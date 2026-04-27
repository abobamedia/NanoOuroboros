"""Telegram student intake, pack state, and feedback learning loop.

This module keeps student-facing Telegram workflow deterministic and cheap:
authorization, Drive intake state, inline callback tokens, append-only feedback,
and the handoff into the existing Direct generator/judge skills.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import secrets
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from ouroboros.utils import append_jsonl, safe_relpath
from supervisor.state import acquire_file_lock, atomic_write_text, release_file_lock


SESSION_TTL_HOURS = 48
CALLBACK_PREFIX = "ou:"
CALLBACK_DATA_LIMIT_BYTES = 64
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_CHUNK_TARGET = 3500

STAGE_IDLE = "IDLE"
STAGE_AWAITING_DRIVE_URL = "AWAITING_DRIVE_URL"
STAGE_DOWNLOADING = "DOWNLOADING"
STAGE_AWAITING_BRIEF = "AWAITING_BRIEF"
STAGE_GENERATING = "GENERATING"
STAGE_AWAITING_FEEDBACK = "AWAITING_FEEDBACK"
STAGE_REASON_SELECT = "REASON_SELECT"
STAGE_REVISING = "REVISING"
STAGE_FINALIZING = "FINALIZING"
STAGE_EXPIRED = "EXPIRED"

REASON_CATEGORIES = {
    "too_generic",
    "not_human",
    "weak_hook",
    "weak_offer",
    "wrong_audience",
    "wrong_awareness_level",
    "untrue_or_risky_claim",
    "moderation_risk",
    "duplicate_angle",
    "not_grounded_in_export",
    "bad_tone",
    "needs_more_specificity",
    "good_but_not_for_this_project",
}

_REASON_LABELS = {
    "too_generic": "Слишком общее",
    "not_human": "Не по-человечески",
    "weak_hook": "Слабый хук",
    "weak_offer": "Слабый оффер",
    "wrong_audience": "Не та аудитория",
    "wrong_awareness_level": "Не тот прогрев",
    "untrue_or_risky_claim": "Рискованное обещание",
    "moderation_risk": "Риск модерации",
    "duplicate_angle": "Дубль угла",
    "not_grounded_in_export": "Не из данных",
    "bad_tone": "Не тот тон",
    "needs_more_specificity": "Нужна конкретика",
    "good_but_not_for_this_project": "Хорошо, но не сюда",
}


@dataclass(frozen=True)
class TelegramResponse:
    text: str
    reply_markup: dict[str, Any] | None = None


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def data_dir() -> pathlib.Path:
    raw = os.environ.get("OUROBOROS_DATA_DIR", "").strip()
    if raw:
        return pathlib.Path(raw).expanduser().resolve()
    return (pathlib.Path.home() / "Ouroboros" / "data").resolve()


def workspace_root() -> pathlib.Path:
    raw = os.environ.get("OUROBOROS_FILE_BROWSER_DEFAULT", "").strip()
    if raw:
        return pathlib.Path(raw).expanduser().resolve()
    ai_workspace = pathlib.Path.home() / "AI" / "ouroboros-workspace"
    if ai_workspace.exists():
        return ai_workspace.resolve()
    return (pathlib.Path.home() / "ouroboros-workspace").resolve()


def students_index_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or data_dir()) / "state" / "students_index.json"


def _index_lock_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return students_index_path(root).with_suffix(".lock")


def feedback_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or workspace_root()) / "domain_memory" / "yandex_direct" / "student_feedback" / "feedback.jsonl"


def tested_creatives_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or workspace_root()) / "domain_memory" / "yandex_direct" / "tested_creatives.jsonl"


def _load_json(path: pathlib.Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        if not path.exists():
            return default
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception:
        return default


def load_students_index(root: pathlib.Path | None = None) -> dict[str, Any]:
    payload = _load_json(students_index_path(root), {"schema_version": 1, "students": {}})
    payload.setdefault("schema_version", 1)
    students = payload.get("students")
    if not isinstance(students, dict):
        payload["students"] = {}
    return payload


def save_students_index(payload: dict[str, Any], root: pathlib.Path | None = None) -> None:
    path = students_index_path(root)
    lock_fd = acquire_file_lock(_index_lock_path(root))
    try:
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        release_file_lock(_index_lock_path(root), lock_fd)


def _parse_chat_id_list(raw: str) -> set[int]:
    ids: set[int] = set()
    for part in re.split(r"[,;\s]+", str(raw or "").strip()):
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            continue
    return ids


def allowed_chat_ids_from_env() -> set[int]:
    return _parse_chat_id_list(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))


def _student_id_for_chat(chat_id: int) -> str:
    digest = hashlib.sha256(str(int(chat_id)).encode("utf-8")).hexdigest()[:8]
    return f"student_{digest}"


def is_student_approved(chat_id: int, *, root: pathlib.Path | None = None) -> bool:
    chat_key = str(int(chat_id or 0))
    if not chat_key or chat_key == "0":
        return False
    index = load_students_index(root)
    entry = (index.get("students") or {}).get(chat_key) or {}
    if bool(entry.get("approved")):
        return True
    return int(chat_id) in allowed_chat_ids_from_env()


def ensure_student(chat_id: int, *, sender_label: str = "", root: pathlib.Path | None = None) -> dict[str, Any]:
    chat_key = str(int(chat_id))
    index = load_students_index(root)
    students = index.setdefault("students", {})
    now = utc_now_iso()
    entry = dict(students.get(chat_key) or {})
    entry.setdefault("student_id", _student_id_for_chat(chat_id))
    entry["approved"] = bool(entry.get("approved")) or int(chat_id) in allowed_chat_ids_from_env()
    entry.setdefault("first_seen_at", now)
    entry["last_seen_at"] = now
    if sender_label and not entry.get("label"):
        entry["label"] = str(sender_label)[:120]
    entry.setdefault("closed_packs_24h", [])
    students[chat_key] = entry
    save_students_index(index, root)
    return entry


def _update_student_current(chat_id: int, current: dict[str, Any] | None, *, root: pathlib.Path | None = None) -> None:
    index = load_students_index(root)
    entry = index.setdefault("students", {}).setdefault(str(int(chat_id)), {
        "student_id": _student_id_for_chat(chat_id),
        "approved": int(chat_id) in allowed_chat_ids_from_env(),
        "first_seen_at": utc_now_iso(),
        "closed_packs_24h": [],
    })
    entry["last_seen_at"] = utc_now_iso()
    if current:
        entry["current"] = current
    else:
        entry.pop("current", None)
    save_students_index(index, root)


def sanitize_slug(value: str, fallback: str = "project") -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    return safe_relpath(text[:80])


def _student_root(student_id: str) -> pathlib.Path:
    sid = sanitize_slug(student_id, "student")
    return workspace_root() / "students" / sid


def _project_root(student_id: str, project_id: str) -> pathlib.Path:
    return _student_root(student_id) / "projects" / sanitize_slug(project_id, "project")


def _new_request_id() -> str:
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"req_{stamp}_{secrets.token_hex(3)}"


def _new_export_id() -> str:
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"export_{stamp}_{secrets.token_hex(3)}"


def _pack_state_path(student_id: str, project_id: str, request_id: str) -> pathlib.Path:
    return _project_root(student_id, project_id) / "pack_states" / f"{sanitize_slug(request_id, 'request')}.json"


def _pack_state_lock(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(".lock")


def load_pack_state(path: pathlib.Path) -> dict[str, Any] | None:
    payload = _load_json(path, {})
    return payload or None


def save_pack_state(state: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(state["pack_state_path"])
    state["last_activity_at"] = utc_now_iso()
    lock_path = _pack_state_lock(path)
    lock_fd = acquire_file_lock(lock_path)
    try:
        atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        release_file_lock(lock_path, lock_fd)
    return path


def _current_pack_state(chat_id: int) -> dict[str, Any] | None:
    index = load_students_index()
    entry = (index.get("students") or {}).get(str(int(chat_id))) or {}
    current = entry.get("current") if isinstance(entry.get("current"), dict) else {}
    path_raw = str(current.get("pack_state_path") or "")
    if not path_raw:
        return None
    return load_pack_state(pathlib.Path(path_raw))


def _is_expired(state: dict[str, Any]) -> bool:
    raw = str(state.get("last_activity_at") or "")
    if not raw:
        return False
    try:
        last = _dt.datetime.fromisoformat(raw)
        if last.tzinfo is None:
            last = last.replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return False
    age = _dt.datetime.now(tz=_dt.timezone.utc) - last
    return age.total_seconds() > SESSION_TTL_HOURS * 3600


def validate_callback_data(callback_data: str) -> str:
    value = str(callback_data or "")
    if not value.startswith(CALLBACK_PREFIX):
        raise ValueError("callback_data must use ou: prefix")
    if len(value.encode("utf-8")) > CALLBACK_DATA_LIMIT_BYTES:
        raise ValueError("callback_data exceeds Telegram 64-byte limit")
    return value


def _register_callback(state: dict[str, Any], payload: dict[str, Any]) -> str:
    callbacks = state.setdefault("callbacks", {})
    for _ in range(8):
        token = secrets.token_urlsafe(8)
        callback_data = f"{CALLBACK_PREFIX}{token}"
        validate_callback_data(callback_data)
        if token not in callbacks:
            stored = dict(payload)
            stored.setdefault("student_id", state.get("student_id"))
            stored.setdefault("request_id", state.get("request_id"))
            stored.setdefault("created_at", utc_now_iso())
            callbacks[token] = stored
            return callback_data
    raise RuntimeError("failed to allocate callback token")


def _send(ctx: Any, chat_id: int, response: TelegramResponse) -> None:
    kwargs = {}
    if response.reply_markup:
        kwargs["reply_markup"] = response.reply_markup
    try:
        ctx.send_with_budget(chat_id, response.text, **kwargs)
    except TypeError:
        ctx.send_with_budget(chat_id, response.text)


def _send_many(ctx: Any, chat_id: int, responses: Iterable[TelegramResponse]) -> None:
    for response in responses:
        _send(ctx, chat_id, response)


def _drive_url(text: str) -> str:
    match = re.search(r"https?://(?:drive\.google\.com|docs\.google\.com)/\S+", text or "")
    return match.group(0).rstrip(".,);]") if match else ""


def _default_project_id(text: str) -> str:
    lowered = str(text or "").lower()
    for marker in ("ниша", "оффер", "проект"):
        match = re.search(marker + r"\s*[:：-]\s*([^\n,.]{4,80})", lowered)
        if match:
            return sanitize_slug(match.group(1), "project")
    return "direct_project_" + _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d")


def _new_session(chat_id: int, student: dict[str, Any], *, project_id: str | None = None) -> dict[str, Any]:
    student_id = str(student["student_id"])
    request_id = _new_request_id()
    export_id = _new_export_id()
    pid = sanitize_slug(project_id or f"direct_project_{_dt.datetime.now(tz=_dt.timezone.utc).strftime('%Y%m%d')}", "project")
    path = _pack_state_path(student_id, pid, request_id)
    state = {
        "schema_version": 1,
        "student_id": student_id,
        "chat_id": int(chat_id),
        "project_id": pid,
        "export_id": export_id,
        "request_id": request_id,
        "stage": STAGE_AWAITING_DRIVE_URL,
        "drive_url": "",
        "brief_text": "",
        "callbacks": {},
        "hypotheses": [],
        "created_at": utc_now_iso(),
        "last_activity_at": utc_now_iso(),
        "pack_state_path": str(path),
        "pack_json_path": str(_project_root(student_id, pid) / "packs" / f"{request_id}.json"),
        "pack_md_path": str(_project_root(student_id, pid) / "packs" / f"{request_id}.md"),
        "export_dir": str(_project_root(student_id, pid) / "exports" / export_id),
        "brief_path": str(_project_root(student_id, pid) / "briefs" / f"{request_id}.json"),
    }
    save_pack_state(state)
    _update_student_current(chat_id, {
        "student_id": student_id,
        "project_id": pid,
        "request_id": request_id,
        "pack_state_path": str(path),
    })
    return state


def handle_telegram_update(msg: dict[str, Any], ctx: Any, *, telegram_owner: bool = False) -> bool:
    """Handle a non-owner Telegram update.

    Returns True when the update was consumed and must not be routed into the
    generic chat agent.
    """
    if telegram_owner:
        return False
    chat_id = int(msg.get("chat_id") or msg.get("telegram_chat_id") or 0)
    text = str(msg.get("text") or "")
    sender_label = str(msg.get("sender_label") or "")
    callback_data = str(msg.get("callback_data") or "")
    if not chat_id:
        return False

    if not is_student_approved(chat_id):
        _send(ctx, chat_id, TelegramResponse(
            "Доступ к учебному боту пока не подключён. Попроси Влада добавить этот Telegram chat_id в список учеников."
        ))
        return True

    student = ensure_student(chat_id, sender_label=sender_label)
    if callback_data:
        _send_many(ctx, chat_id, _handle_callback(chat_id, student, callback_data))
        return True

    lowered = text.strip().lower()
    command = lowered.split(maxsplit=1)[0].split("@", 1)[0] if lowered else ""
    if command in {"/start", "/help"}:
        _send(ctx, chat_id, TelegramResponse(_student_help_text()))
        return True
    if command == "/new" or lowered in {"начать", "новая задача"}:
        state = _new_session(chat_id, student)
        _send(ctx, chat_id, TelegramResponse(
            f"Новая сессия создана: {state['request_id']}.\nПришли Google Drive ссылку на выгрузку из кабинета."
        ))
        return True
    if command == "/cancel":
        state = _current_pack_state(chat_id)
        if state:
            state["stage"] = STAGE_IDLE
            save_pack_state(state)
        _update_student_current(chat_id, None)
        _send(ctx, chat_id, TelegramResponse("Текущая сессия отменена. Для новой задачи напиши /new."))
        return True
    if command == "/status":
        _send(ctx, chat_id, TelegramResponse(_status_text(chat_id)))
        return True
    if command == "/done":
        _send_many(ctx, chat_id, _close_current_pack(chat_id))
        return True
    if command == "/more":
        state = _current_pack_state(chat_id)
        if not state:
            _send(ctx, chat_id, TelegramResponse("Активного пакета нет. Напиши /new."))
            return True
        _send_many(ctx, chat_id, _pack_chunk_responses(state, advance=True))
        return True
    if command == "/delete_current":
        _send_many(ctx, chat_id, _delete_current(chat_id))
        return True
    if command == "/take":
        _send_many(ctx, chat_id, _command_feedback(chat_id, "approved", text))
        return True
    if command == "/skip":
        _send_many(ctx, chat_id, _command_feedback(chat_id, "rejected", text))
        return True
    if command == "/rewrite":
        _send_many(ctx, chat_id, _command_feedback(chat_id, "needs_rewrite", text))
        return True

    state = _current_pack_state(chat_id)
    url = _drive_url(text)
    if state and state.get("stage") == STAGE_REVISING:
        _send_many(ctx, chat_id, _save_rewrite(chat_id, text))
        return True
    if url:
        if state and state.get("stage") not in {STAGE_IDLE, STAGE_EXPIRED}:
            state["drive_url"] = url
        else:
            state = _new_session(chat_id, student, project_id=_default_project_id(text))
            state["drive_url"] = url
        state["stage"] = STAGE_AWAITING_BRIEF
        save_pack_state(state)
        brief = text.replace(url, "").strip(" \n\t-:,.")
        if len(brief) >= 20:
            _send_many(ctx, chat_id, _start_generation_from_brief(state, brief))
        else:
            _send(ctx, chat_id, TelegramResponse(
                "Ссылку принял. Теперь напиши краткий бриф: ниша, оффер, гео, цель, сколько вариантов, что нельзя обещать."
            ))
        return True
    if state and state.get("stage") == STAGE_AWAITING_BRIEF:
        _send_many(ctx, chat_id, _start_generation_from_brief(state, text))
        return True

    _send(ctx, chat_id, TelegramResponse(
        "Пришли Google Drive ссылку на выгрузку или начни новую сессию командой /new."
    ))
    return True


def _student_help_text() -> str:
    return (
        "Я учебный бот для рекламных гипотез Яндекс Директ.\n\n"
        "Как работать:\n"
        "1. /new\n"
        "2. Пришли Google Drive ссылку на выгрузку.\n"
        "3. Напиши бриф: ниша, оффер, гео, цель, аудитория, сколько вариантов.\n"
        "4. Оцени варианты кнопками: взять, отклонить, переписать.\n\n"
        "Команды: /status, /more, /take N, /skip N reason, /rewrite N текст, /done, /cancel."
    )


def _status_text(chat_id: int) -> str:
    state = _current_pack_state(chat_id)
    if not state:
        return "Активной сессии нет. Напиши /new или пришли Google Drive ссылку."
    return (
        f"Сессия: {state.get('request_id')}\n"
        f"Проект: {state.get('project_id')}\n"
        f"Стадия: {state.get('stage')}\n"
        f"Вариантов в пакете: {len(state.get('hypotheses') or [])}"
    )


def _start_generation_from_brief(state: dict[str, Any], brief_text: str) -> list[TelegramResponse]:
    if not brief_text.strip():
        return [TelegramResponse("Бриф пустой. Напиши нишу, оффер, гео и цель.")]
    state["brief_text"] = brief_text.strip()
    state["stage"] = STAGE_GENERATING
    pathlib.Path(state["brief_path"]).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pathlib.Path(state["brief_path"]), json.dumps({
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "student_id": state["student_id"],
        "project_id": state["project_id"],
        "export_id": state["export_id"],
        "request_id": state["request_id"],
        "raw_brief": state["brief_text"],
    }, ensure_ascii=False, indent=2))
    save_pack_state(state)

    try:
        pack = _run_generation(state)
    except Exception as exc:
        state["stage"] = STAGE_AWAITING_BRIEF
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        save_pack_state(state)
        return [TelegramResponse(
            "Не смог собрать пакет автоматически. Ошибка сохранена в состоянии сессии. "
            "Проверь ссылку/формат выгрузки или пришли CSV через Google Drive."
        )]

    state["hypotheses"] = pack["hypotheses"]
    state["generation_meta"] = pack.get("meta", {})
    state["stage"] = STAGE_AWAITING_FEEDBACK
    state["visible_offset"] = 0
    _write_pack_files(state, pack)
    save_pack_state(state)
    return [
        TelegramResponse(
            f"Пакет готов: {len(state['hypotheses'])} вариантов. Показываю первые."
        ),
        *_pack_chunk_responses(state, advance=False),
    ]


def _run_generation(state: dict[str, Any]) -> dict[str, Any]:
    """Run the existing Direct generator/judge skills for one student request."""
    files = _ensure_export_files(state)
    if not files:
        raise ValueError("no csv/tsv/xlsx files found after Drive ingestion")

    workspace = workspace_root()
    pipeline = _load_module("direct_creative_loop_student", workspace / "pipelines" / "direct_creative_loop.py")
    generator = _load_module("direct_ad_generator_student", workspace / "skills" / "direct_ad_generator" / "generator.py")
    judge = _load_module("direct_ad_judge_student", workspace / "skills" / "direct_ad_judge" / "judge.py")

    brief = pipeline.build_creative_brief_from_files(files, source=state.get("drive_url") or state.get("export_dir"))
    known_memory = build_known_memory(state["student_id"], state["project_id"])
    avoid_patterns = known_memory.get("avoid_patterns") or []
    offer = _extract_offer(state.get("brief_text") or "")
    hypotheses, generator_cost = generator.generate_hypotheses(
        brief,
        offer=offer,
        audience_awareness=_extract_awareness(state.get("brief_text") or ""),
        avoid_patterns=avoid_patterns,
        max_retries=1,
    )
    judge_report, judge_cost = judge.evaluate_hypotheses(
        brief,
        hypotheses,
        known_memory=known_memory,
        max_retries=1,
    )
    normalized = _rank_hypotheses(hypotheses, judge_report, limit=30)
    return {
        "hypotheses": normalized,
        "judge_report": judge_report,
        "meta": {
            "generator_cost_usd": generator_cost,
            "judge_cost_usd": judge_cost,
            "source_files": [str(path) for path in files],
            "avoid_patterns": avoid_patterns,
        },
    }


def _load_module(name: str, path: pathlib.Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_export_files(state: dict[str, Any]) -> list[pathlib.Path]:
    export_dir = pathlib.Path(state["export_dir"]).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    if state.get("drive_url") and not any(export_dir.iterdir()):
        from ouroboros.tools.url_ingest import _read_url_impl

        result = _read_url_impl(
            None,  # ToolContext is unused by the implementation.
            state["drive_url"],
            dest_dir=str(export_dir),
            max_bytes=600_000_000,
            timeout_sec=240,
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "read_url failed")
    return sorted(
        path for path in export_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx"}
    )


def _extract_offer(text: str) -> str:
    match = re.search(r"оффер\s*[:：-]\s*([^\n]{3,160})", text.lower())
    return (match.group(1).strip() if match else text.strip()[:120]) or "Рекламный оффер"


def _extract_awareness(text: str) -> str:
    lowered = text.lower()
    if "холод" in lowered or "cold" in lowered:
        return "cold"
    if "горяч" in lowered or "hot" in lowered:
        return "hot"
    return "warm"


def _rank_hypotheses(
    hypotheses: list[dict[str, Any]],
    judge_report: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    evaluations = judge_report.get("evaluations") or []
    by_headline: dict[str, dict[str, Any]] = {}
    for entry in evaluations:
        if not isinstance(entry, dict):
            continue
        hyp = entry.get("hypothesis") or {}
        by_headline[str(hyp.get("headline") or "").strip()] = entry
    ranked: list[dict[str, Any]] = []
    for index, hypothesis in enumerate(hypotheses):
        item = dict(hypothesis)
        item.setdefault("hypothesis_id", f"hyp_{index + 1:04d}")
        item.setdefault("source_index", index)
        evaluation = by_headline.get(str(item.get("headline") or "").strip(), {})
        item["judge_verdict"] = evaluation.get("verdict") or ""
        item["judge_score"] = int(evaluation.get("score") or 0)
        item["test_priority"] = evaluation.get("test_priority") or ""
        ranked.append(item)
    ranked.sort(key=lambda item: (item.get("judge_score", 0), item.get("test_priority") == "high"), reverse=True)
    return _dedup_items(ranked)[:limit]


def _dedup_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = _dedup_headlines_from_memory()
    result = []
    for item in items:
        headline = str(item.get("headline") or "").strip().lower()
        text = str(item.get("text") or "").strip().lower()
        key = f"{headline}\n{text}"
        if not headline or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedup_headlines_from_memory() -> set[str]:
    root = workspace_root() / "domain_memory" / "yandex_direct"
    seen: set[str] = set()
    for event in _iter_jsonl(tested_creatives_path()):
        headline = str(event.get("headline") or "").strip().lower()
        text = str(event.get("text") or "").strip().lower()
        if headline:
            seen.add(f"{headline}\n{text}")
    for path in root.glob("**/ads_pack*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in normalize_pack_items(payload):
            headline = str(item.get("headline") or "").strip().lower()
            text = str(item.get("text") or "").strip().lower()
            if headline:
                seen.add(f"{headline}\n{text}")
    return seen


def normalize_pack_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        candidates: list[Any] = []
        for key in ("hypotheses", "launch_now", "reserve_ab_test", "reserve", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        judge_report = payload.get("judge_report") if isinstance(payload.get("judge_report"), dict) else {}
        evaluations = judge_report.get("evaluations") if isinstance(judge_report, dict) else []
        if isinstance(evaluations, list):
            candidates.extend(entry.get("hypothesis") for entry in evaluations if isinstance(entry, dict))
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []
    normalized = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item.setdefault("hypothesis_id", f"hyp_{index + 1:04d}")
        item.setdefault("source_index", index)
        normalized.append(item)
    return normalized


def _write_pack_files(state: dict[str, Any], pack: dict[str, Any]) -> None:
    pack_json = pathlib.Path(state["pack_json_path"])
    pack_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pack_json, json.dumps({
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "student_id": state["student_id"],
        "project_id": state["project_id"],
        "export_id": state["export_id"],
        "request_id": state["request_id"],
        "hypotheses": pack.get("hypotheses", []),
        "judge_report": pack.get("judge_report", {}),
        "meta": pack.get("meta", {}),
    }, ensure_ascii=False, indent=2))
    lines = [f"# Ads pack {state['request_id']}", ""]
    for idx, item in enumerate(pack.get("hypotheses", []), 1):
        lines.append(f"## {idx}. {item.get('headline', '')}")
        lines.append(str(item.get("text") or ""))
        lines.append(f"- angle: {item.get('angle', '')}")
        lines.append(f"- judge: {item.get('judge_verdict', '')} {item.get('judge_score', '')}")
        lines.append("")
    atomic_write_text(pathlib.Path(state["pack_md_path"]), "\n".join(lines))


def _pack_chunk_responses(state: dict[str, Any], *, advance: bool) -> list[TelegramResponse]:
    hypotheses = list(state.get("hypotheses") or [])
    if not hypotheses:
        return [TelegramResponse("В пакете пока нет вариантов.")]
    offset = int(state.get("visible_offset") or 0)
    if advance:
        offset = min(offset + 3, max(0, len(hypotheses) - 1))
    chunk = hypotheses[offset:offset + 3]
    text_parts = []
    for visible_index, item in enumerate(chunk, offset + 1):
        text_parts.append(
            f"{visible_index}. {item.get('headline', '')}\n"
            f"{item.get('text', '')}\n"
            f"Угол: {item.get('angle', '')}\n"
            f"Оценка judge: {item.get('judge_score', '')}/6 {item.get('judge_verdict', '')}"
        )
    state["visible_offset"] = offset
    reply_markup = _pack_keyboard(state, chunk, offset)
    save_pack_state(state)
    return [TelegramResponse("\n\n".join(text_parts)[:TELEGRAM_CHUNK_TARGET], reply_markup)]


def _pack_keyboard(state: dict[str, Any], chunk: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(chunk, offset + 1):
        hypothesis_id = str(item.get("hypothesis_id") or f"hyp_{index:04d}")
        rows.append([
            {"text": f"Взять {index}", "callback_data": _register_callback(state, {"action": "take", "hypothesis_id": hypothesis_id})},
            {"text": f"Отклонить {index}", "callback_data": _register_callback(state, {"action": "reject", "hypothesis_id": hypothesis_id})},
            {"text": f"Переписать {index}", "callback_data": _register_callback(state, {"action": "rewrite", "hypothesis_id": hypothesis_id})},
        ])
    rows.append([
        {"text": "Еще варианты", "callback_data": _register_callback(state, {"action": "more"})},
        {"text": "Готово", "callback_data": _register_callback(state, {"action": "done"})},
    ])
    return {"inline_keyboard": rows}


def _reason_keyboard(state: dict[str, Any], hypothesis_id: str) -> dict[str, Any]:
    rows = []
    current = []
    for reason, label in _REASON_LABELS.items():
        current.append({
            "text": label,
            "callback_data": _register_callback(state, {
                "action": "reason",
                "hypothesis_id": hypothesis_id,
                "reason": reason,
            }),
        })
        if len(current) == 2:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    rows.append([{"text": "Отмена", "callback_data": _register_callback(state, {"action": "cancel_reason"})}])
    return {"inline_keyboard": rows}


def _handle_callback(chat_id: int, student: dict[str, Any], callback_data: str) -> list[TelegramResponse]:
    try:
        validate_callback_data(callback_data)
    except ValueError:
        return [TelegramResponse("Кнопка устарела или повреждена. Напиши /status.")]
    state = _current_pack_state(chat_id)
    if not state or _is_expired(state):
        if state:
            state["stage"] = STAGE_EXPIRED
            save_pack_state(state)
        return [TelegramResponse("Сессия устарела. Начни новую через /new.")]
    token = callback_data[len(CALLBACK_PREFIX):]
    payload = (state.get("callbacks") or {}).get(token)
    if not isinstance(payload, dict) or payload.get("student_id") != student.get("student_id"):
        return [TelegramResponse("Эта кнопка не относится к твоей текущей сессии.")]
    action = payload.get("action")
    hypothesis_id = str(payload.get("hypothesis_id") or "")
    if action == "take":
        return [_write_feedback_response(chat_id, state, hypothesis_id, "approved", [], "", "", ui_source="callback")]
    if action == "reject":
        state["stage"] = STAGE_REASON_SELECT
        reply_markup = _reason_keyboard(state, hypothesis_id)
        save_pack_state(state)
        return [TelegramResponse("Почему отклоняем?", reply_markup)]
    if action == "reason":
        reason = str(payload.get("reason") or "")
        return [_write_feedback_response(chat_id, state, hypothesis_id, "rejected", [reason], "", "", ui_source="callback")]
    if action == "rewrite":
        state["stage"] = STAGE_REVISING
        state["pending_rewrite_hypothesis_id"] = hypothesis_id
        save_pack_state(state)
        return [TelegramResponse("Напиши свой вариант или направление переписывания одним сообщением.")]
    if action == "more":
        return _pack_chunk_responses(state, advance=True)
    if action == "done":
        return _close_current_pack(chat_id)
    if action == "cancel_reason":
        state["stage"] = STAGE_AWAITING_FEEDBACK
        save_pack_state(state)
        return [TelegramResponse("Ок, отменил выбор причины.")]
    return [TelegramResponse("Неизвестное действие кнопки. Напиши /status.")]


def _find_hypothesis(state: dict[str, Any], hypothesis_id: str) -> dict[str, Any] | None:
    for item in state.get("hypotheses") or []:
        if str(item.get("hypothesis_id") or "") == str(hypothesis_id):
            return item
    return None


def _hypothesis_by_number(state: dict[str, Any], raw_number: str) -> dict[str, Any] | None:
    try:
        index = int(raw_number) - 1
    except ValueError:
        return None
    items = state.get("hypotheses") or []
    return items[index] if 0 <= index < len(items) else None


def _command_feedback(chat_id: int, verdict: str, text: str) -> list[TelegramResponse]:
    state = _current_pack_state(chat_id)
    if not state:
        return [TelegramResponse("Активного пакета нет. Напиши /new.")]
    parts = text.split(maxsplit=2)
    if len(parts) < 2:
        return [TelegramResponse("Укажи номер варианта: /take 3 или /skip 3 too_generic причина.")]
    item = _hypothesis_by_number(state, parts[1])
    if not item:
        return [TelegramResponse("Не нашёл вариант с таким номером.")]
    reason_categories: list[str] = []
    freeform = ""
    suggested = ""
    if verdict == "rejected":
        tail = parts[2] if len(parts) > 2 else ""
        reason_part, _, freeform = tail.partition(" ")
        reason_categories = [r.strip() for r in reason_part.split(",") if r.strip()]
        unknown = [r for r in reason_categories if r not in REASON_CATEGORIES]
        if unknown:
            return [TelegramResponse(
                "Неизвестная причина: " + ", ".join(unknown) + ". Доступные: " + ", ".join(sorted(REASON_CATEGORIES))
            )]
    if verdict == "needs_rewrite":
        suggested = parts[2] if len(parts) > 2 else ""
        freeform = "student rewrite"
    return [_write_feedback_response(
        chat_id,
        state,
        str(item.get("hypothesis_id") or ""),
        verdict,
        reason_categories,
        freeform,
        suggested,
        ui_source="command",
    )]


def _save_rewrite(chat_id: int, text: str) -> list[TelegramResponse]:
    state = _current_pack_state(chat_id)
    if not state:
        return [TelegramResponse("Активного пакета нет.")]
    hypothesis_id = str(state.get("pending_rewrite_hypothesis_id") or "")
    if not hypothesis_id:
        return [TelegramResponse("Не вижу, какой вариант переписываем.")]
    response = _write_feedback_response(
        chat_id,
        state,
        hypothesis_id,
        "needs_rewrite",
        [],
        "student rewrite",
        text.strip(),
        ui_source="callback",
    )
    state["stage"] = STAGE_AWAITING_FEEDBACK
    state.pop("pending_rewrite_hypothesis_id", None)
    save_pack_state(state)
    return [response]


def _write_feedback_response(
    chat_id: int,
    state: dict[str, Any],
    hypothesis_id: str,
    verdict: str,
    reason_categories: list[str],
    freeform_reason: str,
    suggested_rewrite: str,
    *,
    ui_source: str,
) -> TelegramResponse:
    unknown = [reason for reason in reason_categories if reason not in REASON_CATEGORIES]
    if unknown:
        return TelegramResponse("Неизвестная причина: " + ", ".join(unknown))
    item = _find_hypothesis(state, hypothesis_id) or {}
    event = {
        "schema_version": 2,
        "created_at": utc_now_iso(),
        "student_id": state.get("student_id"),
        "project_id": state.get("project_id"),
        "export_id": state.get("export_id"),
        "request_id": state.get("request_id"),
        "hypothesis_id": hypothesis_id,
        "source_index": item.get("source_index"),
        "asset_type": "text_ad",
        "headline": str(item.get("headline") or ""),
        "text": str(item.get("text") or ""),
        "angle": str(item.get("angle") or ""),
        "verdict": verdict,
        "reason_categories": reason_categories,
        "freeform_reason": freeform_reason,
        "suggested_rewrite": suggested_rewrite,
        "reviewer_role": "student",
        "judge_verdict": item.get("judge_verdict"),
        "judge_score": item.get("judge_score"),
        "ui_source": ui_source,
    }
    append_jsonl(feedback_path(), event)
    if verdict in {"approved", "launched", "winner", "loser"}:
        append_jsonl(tested_creatives_path(), _tested_creative_event(event))
    state["stage"] = STAGE_AWAITING_FEEDBACK
    save_pack_state(state)
    if verdict == "approved":
        return TelegramResponse("Принял: вариант отмечен как подходящий.")
    if verdict == "needs_rewrite":
        return TelegramResponse("Записал переписывание. Следующие генерации учтут это как обучающий пример.")
    return TelegramResponse("Отклонение записано. Следующие генерации будут избегать этого паттерна.")


def _tested_creative_event(feedback_event: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": feedback_event["created_at"],
        "id": feedback_event["hypothesis_id"],
        "student_id": feedback_event["student_id"],
        "project_id": feedback_event["project_id"],
        "request_id": feedback_event["request_id"],
        "headline": feedback_event.get("headline", ""),
        "text": feedback_event.get("text", ""),
        "angle": feedback_event.get("angle", ""),
        "metrics": {},
        "verdict": feedback_event.get("verdict", ""),
        "lesson": feedback_event.get("freeform_reason", ""),
    }


def _close_current_pack(chat_id: int) -> list[TelegramResponse]:
    state = _current_pack_state(chat_id)
    if not state:
        return [TelegramResponse("Активного пакета нет.")]
    state["stage"] = STAGE_IDLE
    save_pack_state(state)
    _append_student_profile_summary(state)
    _update_student_current(chat_id, None)
    return [TelegramResponse("Пакет закрыт. Фидбек сохранён и будет учитываться в следующих генерациях.")]


def _delete_current(chat_id: int) -> list[TelegramResponse]:
    state = _current_pack_state(chat_id)
    if not state:
        return [TelegramResponse("Нечего удалять.")]
    for key in ("pack_json_path", "pack_md_path", "brief_path", "pack_state_path"):
        try:
            pathlib.Path(str(state.get(key) or "")).unlink(missing_ok=True)
        except Exception:
            pass
    append_jsonl(_student_root(state["student_id"]) / "student_session_events.jsonl", {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "event": "delete_current",
        "student_id": state["student_id"],
        "project_id": state["project_id"],
        "request_id": state["request_id"],
    })
    _update_student_current(chat_id, None)
    return [TelegramResponse("Текущая незавершённая сессия удалена. История фидбека не изменялась.")]


def _append_student_profile_summary(state: dict[str, Any]) -> None:
    events = [
        event for event in read_feedback_events()
        if event.get("student_id") == state.get("student_id")
        and event.get("project_id") == state.get("project_id")
        and event.get("request_id") == state.get("request_id")
    ]
    total = len(events)
    approved = sum(1 for event in events if event.get("verdict") == "approved")
    rejected = sum(1 for event in events if event.get("verdict") == "rejected")
    reasons: dict[str, int] = {}
    for event in events:
        for reason in event.get("reason_categories") or []:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    profile = _student_root(str(state["student_id"])) / "student_profile.md"
    profile.parent.mkdir(parents=True, exist_ok=True)
    existing = profile.read_text(encoding="utf-8") if profile.exists() else "# Student profile\n"
    line = (
        f"\n## {utc_now_iso()} {state.get('project_id')} {state.get('request_id')}\n"
        f"- feedback_events: {total}\n"
        f"- approve_rate: {approved}/{total if total else 1}\n"
        f"- rejected: {rejected}\n"
        f"- common_rejections: {json.dumps(reasons, ensure_ascii=False, sort_keys=True)}\n"
    )
    atomic_write_text(profile, existing.rstrip() + "\n" + line)


def _iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def read_feedback_events(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    events = []
    for event in _iter_jsonl(path or feedback_path()):
        normalized = dict(event)
        normalized.setdefault("schema_version", 1)
        normalized.setdefault("headline", "")
        normalized.setdefault("text", "")
        normalized.setdefault("angle", "")
        normalized.setdefault("judge_verdict", "")
        normalized.setdefault("judge_score", None)
        normalized.setdefault("source_index", None)
        categories = normalized.get("reason_categories")
        normalized["reason_categories"] = categories if isinstance(categories, list) else []
        events.append(normalized)
    return events


def build_known_memory(student_id: str, project_id: str, *, limit: int = 50) -> dict[str, Any]:
    scoped = [
        event for event in read_feedback_events()
        if event.get("student_id") == student_id and event.get("project_id") == project_id
    ][-limit:]
    reason_counts: dict[str, int] = {}
    recent_rejected = []
    suggested_rewrites = []
    for event in scoped:
        for reason in event.get("reason_categories") or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        if event.get("verdict") in {"rejected", "needs_rewrite", "loser"}:
            headline = str(event.get("headline") or "").strip()
            text = str(event.get("text") or "").strip()
            if headline or text:
                recent_rejected.append({"headline": headline, "text": text, "angle": event.get("angle", "")})
        if event.get("suggested_rewrite"):
            suggested_rewrites.append(str(event.get("suggested_rewrite")))
    avoid_patterns = []
    for reason, count in sorted(reason_counts.items(), key=lambda pair: pair[1], reverse=True):
        if count >= 2:
            avoid_patterns.append(reason)
    for item in recent_rejected[-10:]:
        if item.get("headline"):
            avoid_patterns.append(f"Не повторять отклоненный заголовок: {item['headline']}")
    owner_prefs = workspace_root() / "domain_memory" / "owner_preferences.md"
    return {
        "student_feedback_count": len(scoped),
        "reason_counts": reason_counts,
        "recent_rejected": recent_rejected[-20:],
        "suggested_rewrites": suggested_rewrites[-20:],
        "avoid_patterns": avoid_patterns[:30],
        "owner_style": owner_prefs.read_text(encoding="utf-8") if owner_prefs.exists() else "",
    }
