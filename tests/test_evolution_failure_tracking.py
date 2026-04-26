"""Regression tests for evolution failure tracking."""

import json
from types import SimpleNamespace


class FakeBridge:
    def __init__(self):
        self.pushed = []

    def push_log(self, payload):
        self.pushed.append(payload)


class FakeEvolutionCtx:
    def __init__(self, tmp_path, state, start_sha="aaa"):
        self.DRIVE_ROOT = tmp_path
        self.REPO_DIR = tmp_path
        self.RUNNING = {
            "evo-1": {
                "task": {"id": "evo-1", "type": "evolution"},
                "start_git_sha": start_sha,
            }
        }
        self.WORKERS = {}
        self.bridge = FakeBridge()
        self.saved_states = []
        self._state = dict(state)
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "task_results").mkdir(exist_ok=True)

    def load_state(self):
        return dict(self._state)

    def save_state(self, st):
        self._state = dict(st)
        self.saved_states.append(dict(st))

    @staticmethod
    def append_jsonl(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def persist_queue_snapshot(self, reason=""):
        self.last_snapshot_reason = reason


def _task_done_event(*, cost, rounds):
    return {
        "ts": "2026-04-26T10:00:00Z",
        "task_id": "evo-1",
        "task_type": "evolution",
        "cost_usd": cost,
        "total_rounds": rounds,
        "prompt_tokens": 100,
        "completion_tokens": 10,
    }


def _supervisor_log_lines(tmp_path):
    log_path = tmp_path / "logs" / "supervisor.jsonl"
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_evolution_commit_landed_resets_failures_even_with_zero_cost(tmp_path, monkeypatch):
    from supervisor import events as ev_module

    ctx = FakeEvolutionCtx(tmp_path, {"evolution_consecutive_failures": 2}, start_sha="before")
    monkeypatch.setattr(ev_module, "_get_current_repo_head", lambda _ctx: "after")

    ev_module._handle_task_done(_task_done_event(cost=0.0, rounds=30), ctx)

    assert ctx.saved_states[-1]["evolution_consecutive_failures"] == 0
    assert not [e for e in _supervisor_log_lines(tmp_path) if e.get("type") == "evolution_task_failure_tracked"]


def test_evolution_cost_success_resets_failures_without_commit(tmp_path, monkeypatch):
    from supervisor import events as ev_module

    ctx = FakeEvolutionCtx(tmp_path, {"evolution_consecutive_failures": 2}, start_sha="same")
    monkeypatch.setattr(ev_module, "_get_current_repo_head", lambda _ctx: "same")

    ev_module._handle_task_done(_task_done_event(cost=0.50, rounds=30), ctx)

    assert ctx.saved_states[-1]["evolution_consecutive_failures"] == 0
    assert not [e for e in _supervisor_log_lines(tmp_path) if e.get("type") == "evolution_task_failure_tracked"]


def test_evolution_low_rounds_tracks_failure_reason(tmp_path, monkeypatch):
    from supervisor import events as ev_module

    ctx = FakeEvolutionCtx(tmp_path, {"evolution_consecutive_failures": 1}, start_sha="same")
    monkeypatch.setattr(ev_module, "_get_current_repo_head", lambda _ctx: "same")

    ev_module._handle_task_done(_task_done_event(cost=0.0, rounds=2), ctx)

    assert ctx.saved_states[-1]["evolution_consecutive_failures"] == 2
    failures = [e for e in _supervisor_log_lines(tmp_path) if e.get("type") == "evolution_task_failure_tracked"]
    assert failures[-1]["reason"] == "low_rounds"
    assert failures[-1]["start_git_sha"] == "same"
    assert failures[-1]["end_git_sha"] == "same"
    assert failures[-1]["commit_landed"] is False


def test_evolution_failure_logic_migration_resets_stale_counter_once():
    from supervisor.state import ensure_state_defaults

    migrated = ensure_state_defaults({
        "evolution_consecutive_failures": 3,
        "evolution_failure_logic_version": 1,
    })
    assert migrated["evolution_consecutive_failures"] == 0
    assert migrated["evolution_failure_logic_version"] == 2

    unchanged = ensure_state_defaults({
        "evolution_consecutive_failures": 2,
        "evolution_failure_logic_version": 2,
    })
    assert unchanged["evolution_consecutive_failures"] == 2
    assert unchanged["evolution_failure_logic_version"] == 2


def test_evolution_failure_logic_migration_persists_on_load(tmp_path):
    from supervisor import state as state_module

    (tmp_path / "state").mkdir()
    (tmp_path / "locks").mkdir()
    state_module.init(tmp_path, total_budget_limit=10.0)
    state_module.STATE_PATH.write_text(
        json.dumps({
            "evolution_consecutive_failures": 3,
            "evolution_failure_logic_version": 1,
        }),
        encoding="utf-8",
    )

    loaded = state_module.load_state()
    persisted = json.loads(state_module.STATE_PATH.read_text(encoding="utf-8"))

    assert loaded["evolution_consecutive_failures"] == 0
    assert loaded["evolution_failure_logic_version"] == 2
    assert persisted["evolution_consecutive_failures"] == 0
    assert persisted["evolution_failure_logic_version"] == 2
