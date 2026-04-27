"""Telegram-specific routing helpers for server.py."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ouroboros.student_session import (
    handle_telegram_update,
    load_students_index,
    workspace_root,
)
from ouroboros.telegram_gateway import OWNER_COMMANDS, OWNER_ONLY_TEXT
from supervisor.state import atomic_write_text


@dataclass(frozen=True)
class TelegramRouteResult:
    telegram_source: bool
    telegram_owner: bool
    telegram_command: str
    consumed: bool = False


def route_telegram_control(
    *,
    chat_id: int,
    user_id: int,
    text: str,
    sender_label: str,
    telegram_chat_id: int,
    callback_data: str,
    callback_query_id: str,
    ctx: Any,
    data_dir: pathlib.Path,
) -> TelegramRouteResult:
    telegram_source = int(telegram_chat_id or 0) > 0
    telegram_owner = False
    telegram_command = ""
    if not telegram_source:
        return TelegramRouteResult(False, False, "")

    lowered = text.strip().lower()
    command_head = lowered.split(maxsplit=1)[0] if lowered.strip() else ""
    telegram_command = command_head.split("@", 1)[0]
    owner_chat_id = _owner_chat_id_from_env()
    telegram_owner = bool(owner_chat_id and chat_id == owner_chat_id)

    if telegram_command in OWNER_COMMANDS and not telegram_owner:
        ctx.send_with_budget(chat_id, OWNER_ONLY_TEXT)
        return TelegramRouteResult(True, False, telegram_command, True)

    if telegram_owner and _handle_owner_student_command(
        chat_id=chat_id,
        text=text,
        lowered=lowered,
        ctx=ctx,
        data_dir=data_dir,
    ):
        return TelegramRouteResult(True, True, telegram_command, True)

    if not telegram_owner and handle_telegram_update(
        {
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
            "sender_label": sender_label,
            "telegram_chat_id": telegram_chat_id,
            "callback_data": callback_data,
            "callback_query_id": callback_query_id,
        },
        ctx,
        telegram_owner=False,
    ):
        return TelegramRouteResult(True, False, telegram_command, True)

    return TelegramRouteResult(True, telegram_owner, telegram_command, False)


def _owner_chat_id_from_env() -> int:
    owner_raw = os.environ.get("TELEGRAM_OWNER_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID") or ""
    try:
        return int(str(owner_raw).strip())
    except ValueError:
        return 0


def _handle_owner_student_command(
    *,
    chat_id: int,
    text: str,
    lowered: str,
    ctx: Any,
    data_dir: pathlib.Path,
) -> bool:
    if lowered.startswith("/student_view"):
        _student_view(chat_id, text, ctx, data_dir)
        return True
    if lowered.startswith("/student_inject"):
        _student_inject(chat_id, text, ctx, data_dir)
        return True
    if lowered.startswith("/append_owner_pref"):
        _append_owner_pref(chat_id, text, ctx)
        return True
    return False


def _student_view(chat_id: int, text: str, ctx: Any, data_dir: pathlib.Path) -> None:
    parts = text.split(maxsplit=1)
    needle = parts[1].strip() if len(parts) > 1 else ""
    index = load_students_index(data_dir)
    matches = []
    for raw_chat_id, entry in (index.get("students") or {}).items():
        if not needle or needle in {raw_chat_id, str(entry.get("student_id") or "")}:
            matches.append({
                "chat_id": raw_chat_id,
                "student_id": entry.get("student_id"),
                "approved": bool(entry.get("approved")),
                "label": entry.get("label", ""),
                "last_seen_at": entry.get("last_seen_at", ""),
                "current": entry.get("current", {}),
            })
    ctx.send_with_budget(chat_id, json.dumps(matches[:10], ensure_ascii=False, indent=2))


def _student_inject(chat_id: int, text: str, ctx: Any, data_dir: pathlib.Path) -> None:
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        ctx.send_with_budget(chat_id, "Формат: /student_inject <student_id|chat_id> <текст>")
        return
    needle, inject_text = parts[1], parts[2]
    target_chat_id = 0
    index = load_students_index(data_dir)
    for raw_chat_id, entry in (index.get("students") or {}).items():
        if needle in {raw_chat_id, str(entry.get("student_id") or "")}:
            target_chat_id = int(raw_chat_id)
            break
    if not target_chat_id:
        ctx.send_with_budget(chat_id, "Ученика не нашёл.")
        return
    ctx.send_with_budget(target_chat_id, inject_text)
    ctx.send_with_budget(chat_id, f"Отправлено ученику {needle}.")


def _append_owner_pref(chat_id: int, text: str, ctx: Any) -> None:
    parts = text.split(maxsplit=1)
    pref_text = parts[1].strip() if len(parts) > 1 else ""
    if not pref_text:
        ctx.send_with_budget(chat_id, "Формат: /append_owner_pref <предпочтение>")
        return
    pref_path = workspace_root() / "domain_memory" / "owner_preferences.md"
    current = pref_path.read_text(encoding="utf-8") if pref_path.exists() else "# Owner Preferences\n"
    line = f"\n- {datetime.now(timezone.utc).isoformat()} — {pref_text}\n"
    atomic_write_text(pref_path, current.rstrip() + line)
    ctx.send_with_budget(chat_id, "Предпочтение владельца сохранено.")
