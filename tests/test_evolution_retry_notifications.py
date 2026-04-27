"""Regression tests for evolution/review retry notifications."""

from types import SimpleNamespace


class FakeInQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _run_assign_with_task(monkeypatch, task):
    import supervisor.queue as queue
    import supervisor.state as state_module
    import supervisor.workers as workers

    fake_queue = FakeInQueue()
    fake_worker = workers.Worker(wid=1, proc=SimpleNamespace(is_alive=lambda: True), in_q=fake_queue)
    sent_messages = []

    orig_workers = dict(workers.WORKERS)
    orig_pending = list(workers.PENDING)
    orig_running = dict(workers.RUNNING)
    orig_queue_pending = queue.PENDING
    orig_queue_running = queue.RUNNING

    workers.WORKERS.clear()
    workers.WORKERS[1] = fake_worker
    workers.PENDING[:] = [dict(task)]
    workers.RUNNING.clear()
    queue.PENDING = workers.PENDING
    queue.RUNNING = workers.RUNNING

    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 123, "spent_usd": 0.0})
    monkeypatch.setattr(state_module, "budget_remaining", lambda _st: 1000.0)
    monkeypatch.setattr(workers, "send_with_budget", lambda chat_id, text: sent_messages.append((chat_id, text)))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    try:
        workers.assign_tasks()
        return sent_messages, fake_queue.items, dict(workers.RUNNING)
    finally:
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.PENDING[:] = orig_pending
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        queue.PENDING = orig_queue_pending
        queue.RUNNING = orig_queue_running


def test_evolution_retry_notification_names_attempt_instead_of_repeating_started(monkeypatch):
    messages, queued, running = _run_assign_with_task(
        monkeypatch,
        {"id": "evo-retry", "type": "evolution", "chat_id": 123, "text": "EVOLUTION #11", "_attempt": 2},
    )

    assert queued and queued[0]["id"] == "evo-retry"
    assert running["evo-retry"]["attempt"] == 2
    assert messages == [(123, "🧬 Evolution task evo-retry retrying (attempt 2).")]


def test_evolution_first_attempt_still_says_started(monkeypatch):
    messages, queued, running = _run_assign_with_task(
        monkeypatch,
        {"id": "evo-first", "type": "evolution", "chat_id": 123, "text": "EVOLUTION #11"},
    )

    assert queued and queued[0]["id"] == "evo-first"
    assert running["evo-first"]["attempt"] == 1
    assert messages == [(123, "🧬 Evolution task evo-first started.")]


def test_worker_crash_requeue_increments_attempt_before_next_notification(tmp_path, monkeypatch):
    import supervisor.queue as queue
    import supervisor.workers as workers

    dead_proc = SimpleNamespace(is_alive=lambda: False, exitcode=1)
    alive_queue = FakeInQueue()
    sent_messages = []

    orig_drive = workers.DRIVE_ROOT
    orig_queue_drive = queue.DRIVE_ROOT
    orig_workers = dict(workers.WORKERS)
    orig_pending = list(workers.PENDING)
    orig_running = dict(workers.RUNNING)
    orig_queue_pending = queue.PENDING
    orig_queue_running = queue.RUNNING
    orig_last_spawn = workers._LAST_SPAWN_TIME
    orig_crash_ts = list(workers.CRASH_TS)

    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(exist_ok=True)
    workers.WORKERS.clear()
    workers.WORKERS[1] = workers.Worker(wid=1, proc=dead_proc, in_q=FakeInQueue(), busy_task_id="evo-crash")
    workers.PENDING.clear()
    workers.RUNNING.clear()
    workers.RUNNING["evo-crash"] = {
        "task": {"id": "evo-crash", "type": "evolution", "chat_id": 123, "text": "EVOLUTION #11"},
        "worker_id": 1,
        "attempt": 1,
        "started_at": 1.0,
    }
    workers.CRASH_TS.clear()
    workers._LAST_SPAWN_TIME = 0.0
    queue.PENDING = workers.PENDING
    queue.RUNNING = workers.RUNNING

    monkeypatch.setattr(workers, "respawn_worker", lambda wid: workers.WORKERS.__setitem__(wid, workers.Worker(wid=wid, proc=SimpleNamespace(is_alive=lambda: True), in_q=alive_queue)))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 123, "spent_usd": 0.0})
    monkeypatch.setattr(workers, "send_with_budget", lambda chat_id, text: sent_messages.append((chat_id, text)))

    try:
        workers.ensure_workers_healthy()
        assert workers.PENDING and workers.PENDING[0]["_attempt"] == 2
        assert workers.PENDING[0]["crash_retry_from"] == "evo-crash"

        # The real crash-requeued task should now produce the honest retry notification.
        workers.assign_tasks()
        assert sent_messages == [(123, "🧬 Evolution task evo-crash retrying (attempt 2).")]
    finally:
        workers.DRIVE_ROOT = orig_drive
        queue.DRIVE_ROOT = orig_queue_drive
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.PENDING[:] = orig_pending
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        queue.PENDING = orig_queue_pending
        queue.RUNNING = orig_queue_running
        workers._LAST_SPAWN_TIME = orig_last_spawn
        workers.CRASH_TS[:] = orig_crash_ts


def test_worker_sha_verifier_syncs_stale_state_to_live_head(tmp_path, monkeypatch):
    import supervisor.workers as workers

    synced_states = []
    supervisor_events = []
    original_drive = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(exist_ok=True)

    monkeypatch.setattr(workers, "load_state", lambda: {"current_sha": "oldsha", "current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "sync_current_git_identity", lambda sha, branch="": synced_states.append({"current_sha": sha, "current_branch": branch, "spent_usd": 12.34}) or synced_states[-1])
    monkeypatch.setattr(workers, "_current_repo_identity", lambda: ("newsha", "ouroboros"))
    monkeypatch.setattr(workers, "append_jsonl", lambda path, payload: supervisor_events.append(payload))

    try:
        synced = workers._sync_expected_worker_sha_from_head()
    finally:
        workers.DRIVE_ROOT = original_drive

    assert synced["current_sha"] == "newsha"
    assert synced["spent_usd"] == 12.34
    assert synced_states[-1]["current_sha"] == "newsha"
    assert [e for e in supervisor_events if e.get("type") == "worker_expected_sha_synced"]


def test_worker_sha_sync_failure_skips_verification_instead_of_using_stale_sha(tmp_path, monkeypatch):
    import supervisor.workers as workers

    supervisor_events = []
    original_drive = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(exist_ok=True)

    monkeypatch.setattr(workers, "load_state", lambda: {"current_sha": "oldsha", "current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "sync_current_git_identity", lambda sha, branch="": {"current_sha": "oldsha", "current_branch": "ouroboros"})
    monkeypatch.setattr(workers, "_current_repo_identity", lambda: ("newsha", "ouroboros"))
    monkeypatch.setattr(workers, "append_jsonl", lambda path, payload: supervisor_events.append(payload))

    try:
        synced = workers._sync_expected_worker_sha_from_head()
    finally:
        workers.DRIVE_ROOT = original_drive

    assert synced["current_sha"] == ""
    skipped = [e for e in supervisor_events if e.get("type") == "worker_expected_sha_sync_skipped"]
    assert skipped and skipped[-1]["target_sha"] == "newsha"
    assert skipped[-1]["state_sha"] == "oldsha"


def test_worker_sha_verifier_still_warns_on_real_mismatch_after_sync(tmp_path, monkeypatch):
    import supervisor.workers as workers

    sent_messages = []
    supervisor_events = []
    original_drive = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(exist_ok=True)

    monkeypatch.setattr(
        workers,
        "_sync_expected_worker_sha_from_head",
        lambda: {"current_sha": "live-head", "owner_chat_id": 123},
    )
    monkeypatch.setattr(
        workers,
        "_first_worker_boot_event_since",
        lambda _offset: {"type": "worker_boot", "git_sha": "wrong-head", "pid": 456},
    )
    monkeypatch.setattr(workers, "append_jsonl", lambda path, payload: supervisor_events.append(payload))
    monkeypatch.setattr(workers, "send_with_budget", lambda chat_id, text: sent_messages.append((chat_id, text)))

    try:
        workers._verify_worker_sha_after_spawn(0, timeout_sec=1.0)
    finally:
        workers.DRIVE_ROOT = original_drive

    verify_events = [e for e in supervisor_events if e.get("type") == "worker_sha_verify"]
    assert verify_events and verify_events[-1]["ok"] is False
    assert verify_events[-1]["expected_sha"] == "live-head"
    assert verify_events[-1]["observed_sha"] == "wrong-head"
    assert sent_messages == [(123, "⚠️ Worker SHA mismatch after spawn: expected live-hea, got wrong-he")]
