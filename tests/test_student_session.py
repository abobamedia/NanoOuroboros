import datetime as dt
import json
import pathlib
import sys

import pytest

import ouroboros.student_session as student_session


def _roots(monkeypatch, tmp_path):
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    data.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(workspace))
    return data, workspace


class _Ctx:
    def __init__(self):
        self.sent = []

    def send_with_budget(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))


def test_unknown_student_is_denied_without_state(monkeypatch, tmp_path):
    data, _workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    ctx = _Ctx()

    handled = student_session.handle_telegram_update(
        {"chat_id": 222, "user_id": 222, "text": "/new"},
        ctx,
    )

    assert handled
    assert "пока не подключён" in ctx.sent[0][1]
    index = student_session.load_students_index(data)
    assert index["students"] == {}


def test_approved_student_new_persists_state(monkeypatch, tmp_path):
    _data, workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "333")
    ctx = _Ctx()

    student_session.handle_telegram_update(
        {"chat_id": 333, "user_id": 333, "text": "/new", "sender_label": "Telegram (student)"},
        ctx,
    )

    assert "Новая сессия создана" in ctx.sent[0][1]
    states = list((workspace / "students").glob("*/projects/*/pack_states/*.json"))
    assert len(states) == 1
    payload = json.loads(states[0].read_text(encoding="utf-8"))
    assert payload["stage"] == student_session.STAGE_AWAITING_DRIVE_URL
    assert payload["student_id"].startswith("student_")
    assert not states[0].with_suffix(".lock").exists()


def test_callback_token_expires_and_is_student_scoped(monkeypatch, tmp_path):
    _data, _workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "111 222")
    s1 = student_session.ensure_student(111)
    s2 = student_session.ensure_student(222)
    state = student_session._new_session(111, s1)
    state["hypotheses"] = [{"hypothesis_id": "hyp_0001", "headline": "H", "text": "T"}]
    callback = student_session._register_callback(state, {"action": "take", "hypothesis_id": "hyp_0001"})
    student_session.save_pack_state(state)

    ctx = _Ctx()
    student_session.handle_telegram_update(
        {"chat_id": 222, "user_id": 222, "callback_data": callback},
        ctx,
    )
    assert "не относится" in ctx.sent[0][1] or "устарела" in ctx.sent[0][1]

    state["last_activity_at"] = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=49)
    ).isoformat()
    student_session.atomic_write_text(
        pathlib.Path(state["pack_state_path"]),
        json.dumps(state, ensure_ascii=False),
    )
    ctx2 = _Ctx()
    student_session.handle_telegram_update(
        {"chat_id": 111, "user_id": 111, "callback_data": callback},
        ctx2,
    )
    assert "устарела" in ctx2.sent[0][1]
    assert s2["student_id"] != s1["student_id"]


def test_feedback_v1_readable_and_v2_event_contains_creative_fields(monkeypatch, tmp_path):
    _data, workspace = _roots(monkeypatch, tmp_path)
    ledger = workspace / "domain_memory" / "yandex_direct" / "student_feedback" / "feedback.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"schema_version":1,"student_id":"student_001","project_id":"p","reason_categories":["too_generic"]}\n',
        encoding="utf-8",
    )

    rows = student_session.read_feedback_events(ledger)

    assert rows[0]["schema_version"] == 1
    assert rows[0]["headline"] == ""
    assert rows[0]["text"] == ""

    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "333")
    student = student_session.ensure_student(333)
    state = student_session._new_session(333, student)
    state["stage"] = student_session.STAGE_AWAITING_FEEDBACK
    state["hypotheses"] = [{
        "hypothesis_id": "hyp_0001",
        "source_index": 0,
        "headline": "Меню уже составлено",
        "text": "БЖУ и рецепты в одном плане",
        "angle": "готовое меню",
        "judge_verdict": "APPROVED",
        "judge_score": 6,
    }]
    student_session.save_pack_state(state)

    response = student_session._write_feedback_response(
        333,
        state,
        "hyp_0001",
        "rejected",
        ["too_generic"],
        "слабо",
        "",
        ui_source="command",
    )

    assert "Отклонение записано" in response.text
    events = student_session.read_feedback_events()
    assert events[-1]["schema_version"] == 2
    assert events[-1]["headline"] == "Меню уже составлено"
    assert events[-1]["angle"] == "готовое меню"


def test_skip_unknown_reason_writes_nothing(monkeypatch, tmp_path):
    _data, workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "333")
    student = student_session.ensure_student(333)
    state = student_session._new_session(333, student)
    state["stage"] = student_session.STAGE_AWAITING_FEEDBACK
    state["hypotheses"] = [{"hypothesis_id": "hyp_0001", "headline": "H", "text": "T"}]
    student_session.save_pack_state(state)

    responses = student_session._command_feedback(333, "rejected", "/skip 1 unknown_reason")

    assert "Неизвестная причина" in responses[0].text
    assert not student_session.feedback_path(workspace).exists()


def test_known_memory_builds_avoid_patterns_from_feedback(monkeypatch, tmp_path):
    _data, workspace = _roots(monkeypatch, tmp_path)
    ledger = student_session.feedback_path(workspace)
    ledger.parent.mkdir(parents=True)
    rows = [
        {
            "schema_version": 2,
            "student_id": "student_001",
            "project_id": "p",
            "headline": "План питания для похудения",
            "text": "Общий текст",
            "angle": "generic",
            "verdict": "rejected",
            "reason_categories": ["too_generic"],
        },
        {
            "schema_version": 2,
            "student_id": "student_001",
            "project_id": "p",
            "headline": "Дневник питания",
            "text": "Еще общий текст",
            "angle": "generic",
            "verdict": "rejected",
            "reason_categories": ["too_generic"],
        },
    ]
    ledger.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    memory = student_session.build_known_memory("student_001", "p")

    assert "too_generic" in memory["avoid_patterns"]
    assert any("План питания для похудения" in item for item in memory["avoid_patterns"])


def test_load_module_registers_dataclass_module(monkeypatch, tmp_path):
    module_path = tmp_path / "workspace_dataclass_module.py"
    module_path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class CreativeBrief:\n"
        "    rows_count: int\n",
        encoding="utf-8",
    )
    module_name = "workspace_dataclass_module_for_student_session_test"
    sys.modules.pop(module_name, None)

    module = student_session._load_module(module_name, module_path)

    assert sys.modules[module_name] is module
    assert module.CreativeBrief(728_000).rows_count == 728_000


def test_extract_target_count_from_student_brief():
    assert student_session._extract_target_count("Вариантов сделай 20") == 20
    assert student_session._extract_target_count("нужно 7 заголовков") == 7
    assert student_session._extract_target_count("сделай 999 вариантов") == 30
    assert student_session._extract_target_count("без числа") == 20


def test_multi_tenant_status_does_not_leak_other_pack(monkeypatch, tmp_path):
    _data, _workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "111 222")
    s1 = student_session.ensure_student(111)
    s2 = student_session.ensure_student(222)
    state1 = student_session._new_session(111, s1, project_id="secret_project")
    state1["hypotheses"] = [{"hypothesis_id": "hyp_0001", "headline": "Секретный заголовок"}]
    state1["stage"] = student_session.STAGE_AWAITING_FEEDBACK
    student_session.save_pack_state(state1)
    state2 = student_session._new_session(222, s2, project_id="public_project")
    state2["stage"] = student_session.STAGE_AWAITING_DRIVE_URL
    student_session.save_pack_state(state2)

    status = student_session._status_text(222)

    assert "public_project" in status
    assert "secret_project" not in status
    assert "Секретный заголовок" not in status


def test_callback_data_limit_guard():
    with pytest.raises(ValueError):
        student_session.validate_callback_data("ou:" + "x" * 80)


def test_e2e_student_flow_with_mocked_generation(monkeypatch, tmp_path):
    _data, _workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "333")

    def fake_generation(state):
        return {
            "hypotheses": [{
                "hypothesis_id": "hyp_0001",
                "source_index": 0,
                "headline": "Что есть завтра, чтобы не сорваться",
                "text": "Меню, рецепты и БЖУ уже собраны в один план.",
                "angle": "антисрыв",
                "judge_verdict": "APPROVED",
                "judge_score": 6,
            }],
            "judge_report": {"evaluations": []},
            "meta": {"mocked": True},
        }

    monkeypatch.setattr(student_session, "_run_generation", fake_generation)
    ctx = _Ctx()

    student_session.handle_telegram_update(
        {
            "chat_id": 333,
            "user_id": 333,
            "text": "https://drive.google.com/drive/folders/demo ниша похудение, оффер меню, гео РФ",
            "sender_label": "Telegram (student)",
        },
        ctx,
    )

    assert "Пакет готов" in ctx.sent[0][1]
    reply_markup = ctx.sent[1][2]["reply_markup"]
    reject_button = reply_markup["inline_keyboard"][0][1]["callback_data"]

    ctx2 = _Ctx()
    student_session.handle_telegram_update(
        {"chat_id": 333, "user_id": 333, "callback_data": reject_button},
        ctx2,
    )
    assert "Почему отклоняем" in ctx2.sent[0][1]
    reason_button = ctx2.sent[0][2]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]

    ctx3 = _Ctx()
    student_session.handle_telegram_update(
        {"chat_id": 333, "user_id": 333, "callback_data": reason_button},
        ctx3,
    )

    assert "Отклонение записано" in ctx3.sent[0][1]
    events = student_session.read_feedback_events()
    assert events[-1]["schema_version"] == 2
    assert events[-1]["headline"] == "Что есть завтра, чтобы не сорваться"
    assert events[-1]["ui_source"] == "callback"


def test_retry_reruns_saved_failed_generation(monkeypatch, tmp_path):
    _data, _workspace = _roots(monkeypatch, tmp_path)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "333")
    student = student_session.ensure_student(333)
    state = student_session._new_session(333, student)
    state["drive_url"] = "https://drive.google.com/drive/folders/demo"
    state["brief_text"] = "ниша похудение, оффер дневник питания, гео РФ, 20 вариантов"
    state["stage"] = student_session.STAGE_AWAITING_BRIEF
    state["last_error"] = "AttributeError: old import failure"
    student_session.save_pack_state(state)
    calls = []

    def fake_generation(next_state):
        calls.append(next_state["brief_text"])
        return {
            "hypotheses": [{
                "hypothesis_id": "hyp_0001",
                "source_index": 0,
                "headline": "Минус хаос в питании",
                "text": "Дневник и меню помогают держать дефицит без догадок.",
                "angle": "контроль",
                "judge_verdict": "APPROVED",
                "judge_score": 6,
            }],
            "judge_report": {"evaluations": []},
            "meta": {"mocked": True},
        }

    monkeypatch.setattr(student_session, "_run_generation", fake_generation)
    ctx = _Ctx()

    student_session.handle_telegram_update({"chat_id": 333, "user_id": 333, "text": "/retry"}, ctx)

    assert calls == ["ниша похудение, оффер дневник питания, гео РФ, 20 вариантов"]
    assert "Пакет готов" in ctx.sent[0][1]
    saved = student_session.load_pack_state(pathlib.Path(state["pack_state_path"]))
    assert saved["stage"] == student_session.STAGE_AWAITING_FEEDBACK
