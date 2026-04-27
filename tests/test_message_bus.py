import base64
import json

import supervisor.message_bus as message_bus


def _make_bridge(monkeypatch, settings=None):
    monkeypatch.setattr(message_bus.LocalChatBridge, "_restart_telegram_polling", lambda self: None)
    return message_bus.LocalChatBridge(settings or {})


def test_parse_single_chat_id_valid(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    assert bridge._parse_single_chat_id("12345") == 12345


def test_parse_single_chat_id_empty(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    assert bridge._parse_single_chat_id("") == 0
    assert bridge._parse_single_chat_id("   ") == 0


def test_parse_single_chat_id_invalid(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    assert bridge._parse_single_chat_id("not-a-number") == 0


def test_parse_chat_id_list(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    assert bridge._parse_chat_id_list("111, 222;333 bad") == {111, 222, 333}


def test_telegram_student_intake_allows_multiple_chats_when_enabled(monkeypatch):
    bridge = _make_bridge(monkeypatch, {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_STUDENT_INTAKE_ENABLED": "1",
    })
    bridge._telegram_active_chat_id = 111

    assert bridge._telegram_accepts_chat(111)
    assert bridge._telegram_accepts_chat(222)


def test_telegram_default_keeps_single_active_chat_guard(monkeypatch):
    bridge = _make_bridge(monkeypatch, {"TELEGRAM_BOT_TOKEN": "token"})
    bridge._telegram_active_chat_id = 111

    assert bridge._telegram_accepts_chat(111)
    assert not bridge._telegram_accepts_chat(222)


def test_telegram_accepts_students_index_approved_chat(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    state_dir = data_dir / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "students_index.json").write_text(json.dumps({
        "schema_version": 1,
        "students": {
            "222": {"student_id": "student_222", "approved": True},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data_dir))
    bridge = _make_bridge(monkeypatch, {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_CHAT_ID": "111",
    })

    assert bridge._telegram_accepts_chat(222)


def test_telegram_target_uses_allowed_preferred_chat(monkeypatch):
    bridge = _make_bridge(monkeypatch, {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_CHAT_ID": "111",
        "TELEGRAM_ALLOWED_CHAT_IDS": "222",
    })

    assert bridge._telegram_target(222) == 222
    assert bridge._telegram_target(333) == 111


def test_configure_from_settings_without_extra_fields(monkeypatch):
    """configure_from_settings should work with only TELEGRAM_CHAT_ID."""
    bridge = _make_bridge(monkeypatch)
    bridge.configure_from_settings({
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "999",
    })
    assert bridge._telegram_chat_id == 999
    assert bridge._telegram_active_chat_id == 999


def test_ui_send_enqueues_structured_message_and_broadcasts(monkeypatch):
    bridge = _make_bridge(monkeypatch)
    broadcasts = []
    bridge._broadcast_fn = broadcasts.append

    bridge.ui_send("hello", sender_session_id="sess-1", client_message_id="c-1")
    updates = bridge.get_updates(offset=0, timeout=1)

    assert broadcasts[0]["role"] == "user"
    assert broadcasts[0]["sender_session_id"] == "sess-1"
    assert broadcasts[0]["client_message_id"] == "c-1"
    assert updates[0]["message"]["text"] == "hello"
    assert updates[0]["message"]["source"] == "web"
    assert updates[0]["message"]["sender_session_id"] == "sess-1"
    assert updates[0]["message"]["client_message_id"] == "c-1"


def test_telegram_poll_loop_enqueues_inbound_messages(monkeypatch):
    bridge = _make_bridge(monkeypatch, {"TELEGRAM_BOT_TOKEN": "token"})
    broadcasts = []
    bridge._broadcast_fn = broadcasts.append

    def fake_api(method, **kwargs):
        assert method == "getUpdates"
        bridge._telegram_stop.set()
        return {
            "ok": True,
            "result": [{
                "update_id": 10,
                "message": {
                    "text": "hi from telegram",
                    "chat": {"id": 777},
                    "from": {"id": 888, "username": "anton"},
                },
            }],
        }

    monkeypatch.setattr(bridge, "_telegram_api", fake_api)

    bridge._telegram_stop.clear()
    bridge._telegram_poll_loop()
    updates = bridge.get_updates(offset=0, timeout=1)

    assert updates[0]["message"]["chat"]["id"] == 777
    assert updates[0]["message"]["from"]["id"] == 888
    assert updates[0]["message"]["source"] == "telegram"
    assert updates[0]["message"]["sender_label"] == "Telegram (anton)"
    assert bridge._telegram_active_chat_id == 777
    assert broadcasts[0]["role"] == "user"
    assert broadcasts[0]["source"] == "telegram"


def test_telegram_poll_loop_enqueues_inbound_photo_messages(monkeypatch):
    bridge = _make_bridge(monkeypatch, {"TELEGRAM_BOT_TOKEN": "token"})
    broadcasts = []
    bridge._broadcast_fn = broadcasts.append

    def fake_api(method, **kwargs):
        assert method == "getUpdates"
        bridge._telegram_stop.set()
        return {
            "ok": True,
            "result": [{
                "update_id": 11,
                "message": {
                    "caption": "photo from telegram",
                    "chat": {"id": 777},
                    "from": {"id": 888, "username": "anton"},
                    "photo": [{"file_id": "small"}, {"file_id": "large"}],
                },
            }],
        }

    monkeypatch.setattr(bridge, "_telegram_api", fake_api)
    monkeypatch.setattr(bridge, "_telegram_download_file", lambda file_id, timeout=30: (b"img", "image/png"))

    bridge._telegram_stop.clear()
    bridge._telegram_poll_loop()
    updates = bridge.get_updates(offset=0, timeout=1)

    assert updates[0]["message"]["chat"]["id"] == 777
    assert updates[0]["message"]["text"] == "photo from telegram"
    assert updates[0]["message"]["image_base64"] == base64.b64encode(b"img").decode("ascii")
    assert updates[0]["message"]["image_mime"] == "image/png"
    assert updates[0]["message"]["image_caption"] == "photo from telegram"
    assert broadcasts[0]["type"] == "photo"
    assert broadcasts[0]["role"] == "user"
    assert broadcasts[0]["sender_label"] == "Telegram (anton)"


def test_telegram_bridge_routes_web_messages_replies_actions_and_photos(monkeypatch):
    bridge = _make_bridge(monkeypatch, {
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_CHAT_ID": "555",
    })

    broadcasts = []
    bridge._broadcast_fn = broadcasts.append
    sent_text = []
    sent_actions = []
    sent_photos = []
    monkeypatch.setattr(
        bridge,
        "_send_telegram_text",
        lambda text, preferred_chat_id=0, reply_markup=None: sent_text.append((text, preferred_chat_id, reply_markup)),
    )
    monkeypatch.setattr(
        bridge,
        "_send_telegram_action",
        lambda action, preferred_chat_id=0: sent_actions.append((action, preferred_chat_id)),
    )
    monkeypatch.setattr(
        bridge,
        "_send_telegram_photo",
        lambda photo_bytes, caption="", mime="image/png", preferred_chat_id=0: sent_photos.append(
            (photo_bytes, caption, mime, preferred_chat_id)
        ),
    )

    bridge.ui_send("hello from web", sender_session_id="session-1", client_message_id="c-1")
    updates = bridge.get_updates(offset=0, timeout=1)
    assert updates[0]["message"]["chat"]["id"] == 555
    assert sent_text[0][1] == 555
    assert sent_text[0][0].startswith("WebUI (session-")

    bridge.send_message(555, "assistant reply", task_id="task-42")
    bridge.send_chat_action(555, "typing")
    bridge.send_photo(555, b"img", caption="caption")

    assert sent_text[1] == ("assistant reply", 555, None)
    assert broadcasts[1]["task_id"] == "task-42"
    assert sent_actions == [("typing", 555)]
    assert sent_photos == [(b"img", "caption", "image/png", 555)]
    photo_broadcast = next(item for item in broadcasts if item.get("type") == "photo")
    assert photo_broadcast["ts"].endswith("+00:00")


def test_telegram_bridge_callback_query_and_reply_markup(monkeypatch):
    bridge = _make_bridge(monkeypatch, {"TELEGRAM_BOT_TOKEN": "token"})
    broadcasts = []
    sent_params = []
    bridge._broadcast_fn = broadcasts.append

    def fake_api(method, **kwargs):
        if method == "getUpdates":
            bridge._telegram_stop.set()
            return {
                "ok": True,
                "result": [{
                    "update_id": 12,
                    "callback_query": {
                        "id": "cb-1",
                        "data": "ou:abc",
                        "message": {"chat": {"id": 777}},
                        "from": {"id": 888, "username": "anton"},
                    },
                }],
            }
        sent_params.append((method, kwargs.get("params")))
        return {"ok": True, "result": {}}

    monkeypatch.setattr(bridge, "_telegram_api", fake_api)

    bridge._telegram_stop.clear()
    bridge._telegram_poll_loop()
    updates = bridge.get_updates(offset=0, timeout=1)

    callback = updates[0]["callback_query"]
    assert callback["id"] == "cb-1"
    assert callback["data"] == "ou:abc"
    assert callback["telegram_chat_id"] == 777
    assert callback["source"] == "telegram"
    assert broadcasts[0]["content"] == "[callback] ou:abc"

    markup = {"inline_keyboard": [[{"text": "Взять", "callback_data": "ou:abc"}]]}
    bridge.send_message(777, "выбери", reply_markup=markup)
    assert sent_params[0][0] == "sendMessage"
    assert '"callback_data": "ou:abc"' in sent_params[0][1]["reply_markup"]
