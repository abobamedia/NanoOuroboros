"""Regression coverage for OpenAI-compatible token usage budget tracking."""

import json
import queue
from unittest.mock import patch


def test_openai_compatible_gpt55_estimates_nonzero_cost_from_usage_tokens():
    from ouroboros.pricing import estimate_cost

    cost = estimate_cost(
        "openai-compatible/gpt-5.5",
        prompt_tokens=18,
        completion_tokens=23,
    )

    assert cost > 0
    assert cost == estimate_cost(
        "openai/gpt-5.5",
        prompt_tokens=18,
        completion_tokens=23,
    )


def test_openai_compatible_usage_event_increments_spent_usd(tmp_path):
    from ouroboros.pricing import emit_llm_usage_event, estimate_cost
    from supervisor import events as ev_module
    from supervisor import state as state_module

    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "locks").mkdir()
    state_module.init(tmp_path, total_budget_limit=10.0)
    state_module.save_state({"spent_usd": 0.0, "spent_calls": 0})

    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 0}
    cost = estimate_cost("openai-compatible/gpt-5.5", **usage)
    assert cost > 0

    q = queue.Queue()
    emit_llm_usage_event(
        q,
        task_id="budget-test",
        model="openai-compatible/gpt-5.5",
        usage=usage,
        cost=cost,
        category="task",
        provider="openai-compatible",
        source="loop",
    )
    evt = q.get_nowait()

    class FakeCtx:
        DRIVE_ROOT = tmp_path

        @staticmethod
        def update_budget_from_usage(budget_usage):
            with patch.object(state_module, "check_openrouter_ground_truth", return_value=None):
                state_module.update_budget_from_usage(budget_usage)

    ev_module._handle_llm_usage(evt, FakeCtx())

    st = state_module.load_state()
    assert st["spent_usd"] == cost
    assert st["spent_calls"] == 1
    assert st["spent_tokens_prompt"] == 1000
    assert st["spent_tokens_completion"] == 500

    written = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").strip())
    assert written["provider"] == "openai-compatible"
    assert written["api_key_type"] == "openai-compatible"
    assert written["cost"] == cost
    assert written["cost_estimated"] is True
