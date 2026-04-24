"""Claude Code advisory pre-review gate.

Runs a read-only Claude Code review of the current worktree BEFORE the unified
multi-model pre-commit review. Advisory findings are non-blocking by themselves;
only the *absence* of a fresh matching advisory run blocks repo_commit.

Correct workflow:
  1. Finish ALL edits first
  2. advisory_pre_review(commit_message="...")   ← run AFTER all edits are done
  3. repo_commit(commit_message="...")           ← run IMMEDIATELY after advisory

⚠️ Any edit (repo_write / str_replace_editor) after step 2 automatically marks
   the advisory as stale — you must re-run advisory_pre_review before repo_commit.

Tool surface:
  advisory_pre_review   run a fresh advisory review
  review_status         show advisory history, open obligations, staleness state
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import subprocess
from typing import List, Optional

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.review_state import (
    AdvisoryRunRecord,
    AdvisoryReviewState,
    compute_snapshot_hash,
    format_status_section,
    load_state,
    make_repo_key,
    save_state,
    update_state,
    _utc_now,
)
from ouroboros.tools.advisory_prompt import (
    _build_blocking_history_section,
    build_advisory_prompt as _build_advisory_prompt,
)
from ouroboros.tools.review_helpers import (
    build_advisory_changed_context,
    check_worktree_version_sync as _check_worktree_version_sync_shared,
    get_advisory_runtime_diagnostics as _get_runtime_diagnostics,
    format_advisory_sdk_error as _format_advisory_error,
)
from ouroboros.utils import (
    append_jsonl,
    utc_now_iso,
    truncate_review_artifact as _truncate_review_artifact,
    truncate_review_reason as _truncate_review_reason,
)

log = logging.getLogger(__name__)

_MAX_DIFF_CHARS_ERROR = 500_000  # Fail loudly above this — split the commit
# Advisory prompt budget gate — non-blocking skip when prompt too large (mirrors scope review)
# Claude Code has a 1M token context; 1.6M chars ≈ 400K tokens leaves healthy headroom.
_ADVISORY_PROMPT_MAX_CHARS = 1_600_000  # ~400K tokens
_OBLIGATION_SUFFIX_RE = re.compile(r"\s*\(obligation\s+([a-f0-9]+)\)\s*$", re.IGNORECASE)


def _get_staged_diff(
    repo_dir: pathlib.Path,
    paths: list[str] | None = None,
) -> str:
    """Return staged diff + unstaged diff — full, no truncation.

    When ``paths`` is provided, only diffs for those specific paths are fetched,
    so a large unrelated worktree change cannot trigger the 500K hard-fail or
    bias the advisory review.

    If the combined diff exceeds _MAX_DIFF_CHARS_ERROR (500K chars), returns an
    explicit error message telling the agent to split the commit.
    """
    try:
        path_args = (["--"] + list(paths)) if paths else []
        staged_result = subprocess.run(
            ["git", "diff", "--cached"] + path_args,
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if staged_result.returncode != 0:
            err = (staged_result.stderr or "").strip()[:200]
            return (
                f"⚠️ ADVISORY_ERROR: git diff --cached exited {staged_result.returncode}: {err}"
            )
        unstaged_result = subprocess.run(
            ["git", "diff"] + path_args,
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if unstaged_result.returncode != 0:
            err = (unstaged_result.stderr or "").strip()[:200]
            return (
                f"⚠️ ADVISORY_ERROR: git diff exited {unstaged_result.returncode}: {err}"
            )
        combined = ((staged_result.stdout or "") + (unstaged_result.stdout or "")).strip()
        if len(combined) > _MAX_DIFF_CHARS_ERROR:
            return (
                f"⚠️ ADVISORY_ERROR: staged diff is too large ({len(combined):,} chars). "
                "Split the commit into smaller pieces."
            )
        return combined or "(no unstaged/staged changes found)"
    except Exception as exc:
        return f"⚠️ ADVISORY_ERROR: failed to retrieve diff: {exc}"


def _get_changed_file_list(
    repo_dir: pathlib.Path,
    paths: list[str] | None = None,
) -> str:
    """Return porcelain status, optionally scoped to ``paths``.

    When ``paths`` is provided, only those paths are listed so the advisory
    reviewer is not distracted by unrelated changed files.
    """
    try:
        path_args = (["--"] + list(paths)) if paths else []
        result = subprocess.run(
            ["git", "status", "--porcelain"] + path_args,
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip()[:200]
            return f"⚠️ ADVISORY_ERROR: git status exited {result.returncode}: {err}"
        lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
        return "\n".join(lines) if lines else "(clean — no changed files)"
    except Exception as exc:
        return f"⚠️ ADVISORY_ERROR: git status error: {exc}"


def _prepare_advisory_prompt_context(
    repo_dir: pathlib.Path,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    drive_root: Optional[pathlib.Path] = None,
) -> tuple[str, list[str], str] | str:
    # Fetch diff and changed-file list exactly once, validate, then pass into prompt builder.
    diff_text = _get_staged_diff(repo_dir, paths=paths)
    if diff_text.startswith("⚠️ ADVISORY_ERROR:"):
        return diff_text

    changed_files_text = _get_changed_file_list(repo_dir, paths=paths)
    if changed_files_text.startswith("⚠️ ADVISORY_ERROR:"):
        return changed_files_text

    # Parse touched paths from porcelain output to avoid a second git-status call inside
    # build_touched_file_pack.  Lines are "XY filename" or "(clean — no changed files)".
    try:
        always_inlined = {"docs/ARCHITECTURE.md"}
        resolved_paths, touched_pack, omitted_paths = build_advisory_changed_context(
            repo_dir,
            changed_files_text=changed_files_text,
            paths=paths,
            exclude_paths=always_inlined,
        )
        prompt = _build_advisory_prompt(
            repo_dir,
            commit_message,
            goal=goal,
            scope=scope,
            resolved_paths=resolved_paths,
            drive_root=drive_root,
            diff=diff_text,
            changed_files=changed_files_text,
            touched_pack=touched_pack,
            omitted_paths=omitted_paths,
        )
    except RuntimeError as exc:
        return f"⚠️ ADVISORY_ERROR: failed to build advisory prompt: {exc}"
    except Exception as exc:
        return f"⚠️ ADVISORY_ERROR: unexpected error building prompt: {exc}"

    return prompt, resolved_paths, changed_files_text


def _run_claude_advisory(
    repo_dir: pathlib.Path,
    commit_message: str,
    ctx: ToolContext,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    drive_root: Optional[pathlib.Path] = None,
) -> tuple[list, str]:
    """Run the advisory review via Claude Agent SDK (read-only).

    Returns (items, raw_result). raw_result starts with ⚠️ ADVISORY_ERROR: on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [], "⚠️ ADVISORY_ERROR: ANTHROPIC_API_KEY not set."

    # Resolve model — single source of truth, honours CLAUDE_CODE_MODEL setting
    from ouroboros.gateways.claude_code import resolve_claude_code_model
    model = resolve_claude_code_model()

    prepared = _prepare_advisory_prompt_context(
        repo_dir,
        commit_message,
        goal=goal,
        scope=scope,
        paths=paths,
        drive_root=drive_root,
    )
    if isinstance(prepared, str):
        return [], prepared
    prompt, resolved_paths, _changed_files_text = prepared

    prompt_chars = len(prompt)
    diag = _get_runtime_diagnostics(model, prompt_chars, resolved_paths)

    # Budget gate: non-blocking skip when prompt too large (mirrors scope review)
    if prompt_chars > _ADVISORY_PROMPT_MAX_CHARS:
        tokens_approx = max(1, prompt_chars // 4)
        warning = (
            f"⚠️ ADVISORY_SKIPPED: advisory prompt too large "
            f"({prompt_chars:,} chars, ~{tokens_approx:,} tokens > "
            f"{_ADVISORY_PROMPT_MAX_CHARS:,} char limit). "
            f"Advisory review skipped — non-blocking. Consider splitting the commit."
        )
        log.warning("Advisory skipped — prompt too large: %d chars", prompt_chars)
        return [], warning

    log.info(
        "Advisory SDK call: model=%s prompt_chars=%d touched=%s sdk=%s cli=%s",
        diag["model"], diag["prompt_chars"], diag["touched_paths"],
        diag["sdk_version"], diag["cli_version"],
    )

    try:
        from ouroboros.gateways.claude_code import (
            DEFAULT_CLAUDE_CODE_MAX_TURNS,
            run_readonly,
        )

        result = run_readonly(
            prompt=prompt,
            cwd=str(repo_dir),
            model=model,
            max_turns=DEFAULT_CLAUDE_CODE_MAX_TURNS,
        )

        if not result.success:
            err_msg = _format_advisory_error(
                prefix="SDK/CLI returned failure",
                result_error=result.error,
                stderr_tail=result.stderr_tail,
                session_id=result.session_id,
                diag=diag,
            )
            log.error("Advisory SDK failure:\n%s", err_msg)
            return [], err_msg

        raw_text = result.result_text
        items = _parse_advisory_output(raw_text)
        return items, raw_text

    except ImportError:
        return [], (
            "⚠️ ADVISORY_ERROR: claude-agent-sdk not installed. "
            "Install: pip install 'ouroboros[claude-sdk]'"
        )
    except Exception as e:
        err_msg = _format_advisory_error(
            prefix=f"SDK call raised {type(e).__name__}",
            result_error=str(e),
            stderr_tail="",
            session_id="",
            diag=diag,
        )
        log.error("Advisory SDK exception:\n%s", err_msg)
        return [], err_msg


def _run_llm_advisory(
    repo_dir: pathlib.Path,
    commit_message: str,
    ctx: ToolContext,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    drive_root: Optional[pathlib.Path] = None,
) -> tuple[list, str]:
    """Run advisory review through the configured openai-compatible LLM endpoint."""
    compatible_key = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
    legacy_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
    legacy_base_url = str(os.environ.get("OPENAI_BASE_URL", "") or "").strip()
    if not (compatible_key or (legacy_key and legacy_base_url)):
        return [], "⚠️ ADVISORY_ERROR: no openai-compatible advisory provider configured."

    model = "openai-compatible::gpt-5.4"
    for env_key in ("OUROBOROS_ADVISORY_REVIEW_MODEL", "OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_MODEL"):
        configured_model = str(os.environ.get(env_key, "") or "").strip()
        if configured_model.startswith("openai-compatible::"):
            model = configured_model
            break
    prepared = _prepare_advisory_prompt_context(
        repo_dir,
        commit_message,
        goal=goal,
        scope=scope,
        paths=paths,
        drive_root=drive_root,
    )
    if isinstance(prepared, str):
        return [], prepared
    prompt, resolved_paths, _changed_files_text = prepared

    prompt_chars = len(prompt)
    if prompt_chars > _ADVISORY_PROMPT_MAX_CHARS:
        tokens_approx = max(1, prompt_chars // 4)
        warning = (
            f"⚠️ ADVISORY_SKIPPED: advisory prompt too large "
            f"({prompt_chars:,} chars, ~{tokens_approx:,} tokens > "
            f"{_ADVISORY_PROMPT_MAX_CHARS:,} char limit). "
            f"Advisory review skipped — non-blocking. Consider splitting the commit."
        )
        log.warning("OpenAI-compatible advisory skipped — prompt too large: %d chars", prompt_chars)
        return [], warning

    log.info(
        "OpenAI-compatible advisory call: model=%s prompt_chars=%d touched=%s",
        model, prompt_chars, len(resolved_paths),
    )

    try:
        from ouroboros.config import resolve_effort
        from ouroboros.llm import LLMClient

        msg, usage = LLMClient().chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an advisory code reviewer. Return ONLY a JSON array "
                        "matching the requested schema. No markdown fences, no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            reasoning_effort=resolve_effort("review"),
            max_tokens=4096,
            tool_choice="none",
            temperature=0,
        )
        if usage:
            try:
                from ouroboros.pricing import emit_llm_usage_event, estimate_cost
                from supervisor.state import update_budget_from_usage

                cost = float(usage.get("cost") or 0.0)
                if cost == 0.0:
                    cost = estimate_cost(
                        str(usage.get("resolved_model") or model),
                        int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0),
                        int(usage.get("cached_tokens") or 0),
                        int(usage.get("cache_write_tokens") or 0),
                    )
                event_queue = getattr(ctx, "event_queue", None)
                if event_queue is not None:
                    emit_llm_usage_event(
                        event_queue,
                        str(getattr(ctx, "task_id", "") or ""),
                        model,
                        usage,
                        cost,
                        category="review",
                        provider="openai-compatible",
                        source="advisory_pre_review",
                    )
                else:
                    update_budget_from_usage(usage)
            except Exception:
                log.debug("Failed to emit advisory LLM usage event", exc_info=True)

        content = msg.get("content") if isinstance(msg, dict) else ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item or ""))
            raw_text = "\n".join(part for part in parts if part).strip()
        else:
            raw_text = str(content or "").strip()
        items = _parse_advisory_output(raw_text)
        return items, raw_text
    except Exception as exc:
        err_msg = f"⚠️ ADVISORY_ERROR: openai-compatible advisory failed: {type(exc).__name__}: {exc}"
        log.error(err_msg, exc_info=True)
        return [], err_msg


def _parse_advisory_output(stdout: str) -> list:
    """Extract the JSON findings array from Claude CLI output."""
    # Try direct parse first
    text = stdout.strip()

    # Unwrap Claude Code JSON envelope: {"result": "...", ...}
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and "result" in outer:
            text = str(outer["result"]).strip()
        elif isinstance(outer, list):
            return outer
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse of the inner result
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find embedded JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return []


# -- Audit logging --

def _audit_bypass(ctx: ToolContext, snapshot_hash: str, commit_message: str,
                  bypass_reason: str, task_id: str) -> None:
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "advisory_pre_review_bypassed",
            "snapshot_hash": snapshot_hash,
            "commit_message": commit_message,  # full — no [:200] truncation
            "bypass_reason": bypass_reason,
            "task_id": task_id,
        })
    except Exception:
        pass


def _record_bypass(ctx: ToolContext, state: "AdvisoryReviewState", snapshot_hash: str,
                   commit_message: str, reason: str, task_id: str,
                   drive_root: pathlib.Path,
                   snapshot_paths: Optional[List[str]] = None) -> str:
    """Audit, record, and save a bypassed advisory run. Returns JSON response."""
    _audit_bypass(ctx, snapshot_hash, commit_message, reason, task_id)
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))

    def _mutate(bypass_state: "AdvisoryReviewState") -> None:
        next_run_attempt = len(
            bypass_state.filter_advisory_runs(
                repo_key=repo_key,
                tool_name="advisory_pre_review",
                task_id=task_id,
            )
        ) + 1
        bypass_state.add_run(AdvisoryRunRecord(
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            status="bypassed",
            ts=_utc_now(),
            bypass_reason=reason,
            bypassed_by_task=task_id,
            snapshot_paths=snapshot_paths,
            repo_key=repo_key,
            tool_name="advisory_pre_review",
            task_id=task_id,
            attempt=next_run_attempt,
        ))

    update_state(drive_root, _mutate)
    if "No advisory provider configured" in reason:
        msg = (
            "⚠️ No advisory provider is configured — advisory review skipped automatically. "
            "Bypass has been durably audited in events.jsonl. "
            "Set ANTHROPIC_API_KEY or openai-compatible credentials to enable advisory reviews."
        )
    elif "ANTHROPIC_API_KEY" in reason:
        msg = (
            "⚠️ ANTHROPIC_API_KEY is not set — advisory review skipped automatically. "
            "Bypass has been durably audited in events.jsonl. "
            "Set ANTHROPIC_API_KEY in Settings to enable Claude Code advisory reviews."
        )
    else:
        msg = "Advisory review bypassed. Bypass has been durably audited."
    return json.dumps({"status": "bypassed", "snapshot_hash": snapshot_hash,
                       "bypass_reason": reason, "message": msg},
                      ensure_ascii=False, indent=2)


def _resolve_matching_obligations(
    state: "AdvisoryReviewState",
    items: list,
    snapshot_hash: str,
    *,
    repo_key: str | None = None,
) -> None:
    """Resolve open obligations whose checklist item appears in PASS but NOT in FAIL.

    An obligation is only resolved when the advisory emits PASS for that item
    and does not also emit a contradictory FAIL for the same item.  Conflicting
    entries (both PASS and FAIL for the same item) leave the obligation open so
    the agent is forced to re-examine and produce a clean, unambiguous result.
    """
    if not items:
        return
    # Build per-item verdict sets to detect contradictions
    item_verdicts: dict[str, set[str]] = {}
    obligation_verdicts: dict[str, set[str]] = {}
    for i in items:
        if not isinstance(i, dict):
            continue
        verdict = str(i.get("verdict", "")).upper().strip()
        item_name = str(i.get("item", "")).strip()
        if not item_name or not verdict:
            continue
        explicit_obligation_id = str(i.get("obligation_id", "")).strip().lower()
        match = _OBLIGATION_SUFFIX_RE.search(item_name)
        normalized_item_name = _OBLIGATION_SUFFIX_RE.sub("", item_name).strip().lower()
        if normalized_item_name:
            item_verdicts.setdefault(normalized_item_name, set()).add(verdict)
        if explicit_obligation_id:
            obligation_verdicts.setdefault(explicit_obligation_id, set()).add(verdict)
        if match:
            obligation_verdicts.setdefault(match.group(1).lower(), set()).add(verdict)

    # Only PASS items that have no FAIL entry for the same item
    unambiguous_pass = {
        item_name
        for item_name, verdicts in item_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }
    unambiguous_pass_ids = {
        obligation_id
        for obligation_id, verdicts in obligation_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }

    open_obs = state.get_open_obligations(repo_key=repo_key)

    # Count open obligations per item so item-name fallback is safe only when
    # unambiguous (exactly one open obligation for that item).  With per-finding
    # fingerprint keying, a same-item PASS must not clear a different finding
    # that was not addressed.
    from collections import Counter as _Counter
    item_open_count = _Counter(o.item.lower() for o in open_obs)

    resolved = [
        o.obligation_id for o in open_obs
        if o.obligation_id.lower() in unambiguous_pass_ids
        or (
            o.item.lower() in unambiguous_pass
            and item_open_count[o.item.lower()] == 1
        )
    ]
    if resolved:
        state.resolve_obligations(
            resolved,
            resolved_by=f"advisory run {snapshot_hash[:12]}",
            repo_key=repo_key,
        )


def _next_step_guidance(latest: Optional["AdvisoryRunRecord"], state: "AdvisoryReviewState",
                        stale_from_edit: bool, stale_from_edit_ts: Optional[str],
                        open_obs: list, effective_is_fresh: bool = False) -> str:
    """Return a concrete next-step string based on current advisory state."""
    if not effective_is_fresh:
        # parse_failure for the exact current snapshot (advisory ran but output was unparseable)
        if latest and latest.status == "parse_failure" and not stale_from_edit:
            return (
                "Last advisory run produced unparseable output (parse_failure). "
                "Re-run: advisory_pre_review(commit_message='...'), "
                "or bypass: repo_commit(skip_advisory_pre_review=True) (audited)."
            )
        if stale_from_edit:
            return (
                f"Advisory was invalidated by a worktree edit at {stale_from_edit_ts}. "
                "Complete ALL remaining edits, then run: "
                "advisory_pre_review(commit_message='...')"
            )
        if not state.runs:
            return "No advisory run yet. Run: advisory_pre_review(commit_message='...')"
        return "Advisory is stale (snapshot changed). Run: advisory_pre_review(commit_message='...')"

    # Advisory is effectively fresh — check obligations and findings
    if open_obs:
        return (
            f"Advisory is current but {len(open_obs)} open obligation(s) remain from "
            "previous blocking rounds. repo_commit will be blocked until obligations are "
            "cleared. Fix the issues, re-run advisory_pre_review so it marks them PASS, "
            "or bypass: repo_commit(skip_advisory_pre_review=True) (audited)."
        )

    if latest and latest.status == "skipped":
        return (
            "Advisory was skipped — prompt exceeded the budget gate (prompt too large for advisory). "
            "repo_commit may proceed. Consider splitting the commit into smaller chunks "
            "so advisory can run on the next change."
        )

    if latest and latest.status == "bypassed":
        return (
            "Advisory was bypassed (audited). "
            "No open obligations — repo_commit should proceed. "
            "Consider running advisory_pre_review for a proper review."
        )

    fresh_critical = [
        i for i in (latest.items if latest else []) or []
        if isinstance(i, dict) and str(i.get("verdict", "")).upper() == "FAIL"
        and str(i.get("severity", "")).lower() == "critical"
    ]
    if fresh_critical:
        return (
            f"Advisory found {len(fresh_critical)} critical issue(s). "
            "Fix ALL critical findings, then re-run advisory_pre_review. "
            "Do NOT call repo_commit until advisory is fresh with 0 critical findings."
        )
    return (
        "Advisory is fresh with no critical findings. "
        "Proceed with: repo_commit(commit_message='...'). "
        "⚠️ Do NOT make any further edits — any edit will make advisory stale."
    )


def _check_worktree_version_sync(repo_dir: pathlib.Path) -> str:
    """Worktree version-sync preflight for advisory path (non-fatal, non-blocking).

    Delegates to the shared helper in review_helpers.  The local name is kept as
    a backward-compatible alias so existing tests importing from this module continue
    to work without changes.
    """
    return _check_worktree_version_sync_shared(repo_dir)


# -- Tool handlers --

def _handle_advisory_pre_review(
    ctx: ToolContext,
    commit_message: str = "",
    skip_advisory_pre_review: bool = False,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
) -> str:
    """Run an advisory pre-commit review via Claude Agent SDK (read-only).

    Stores result in durable state so repo_commit can verify freshness.
    Returns structured JSON with findings + status.
    """
    repo_dir = pathlib.Path(ctx.repo_dir)
    drive_root = pathlib.Path(ctx.drive_root)
    repo_key = make_repo_key(repo_dir)

    snapshot_hash = compute_snapshot_hash(repo_dir, commit_message, paths=paths)
    state = load_state(drive_root)
    task_id = str(getattr(ctx, "task_id", "") or "")

    # Handle explicit bypass
    if skip_advisory_pre_review:
        return _record_bypass(ctx, state, snapshot_hash, commit_message,
                               "explicit skip_advisory_pre_review=True", task_id, drive_root,
                               snapshot_paths=paths)

    has_anthropic_advisory = bool(str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip())
    compatible_key = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
    legacy_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
    legacy_base_url = str(os.environ.get("OPENAI_BASE_URL", "") or "").strip()
    has_llm_advisory = bool(compatible_key or (legacy_key and legacy_base_url))
    if not has_anthropic_advisory and not has_llm_advisory:
        reason = "No advisory provider configured — ANTHROPIC_API_KEY and openai-compatible credentials are absent"
        ctx.emit_progress_fn(f"⚠️ Advisory pre-review skipped: {reason}")
        return _record_bypass(ctx, state, snapshot_hash, commit_message,
                               f"{reason} — auto-bypassed", task_id, drive_root,
                               snapshot_paths=paths)

    # Check if we already have a fresh run for this snapshot.
    # BUT: if there are open obligations from a blocked commit, force a re-run
    # even on the same snapshot hash so obligations are explicitly verified.
    existing = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    open_obligations = state.get_open_obligations(repo_key=repo_key)
    if existing and existing.status in ("fresh", "bypassed", "skipped") and not open_obligations:
        return json.dumps({
            "status": "already_fresh",
            "snapshot_hash": snapshot_hash,
            "ts": existing.ts,
            "items": existing.items,
            "message": "A fresh advisory run already exists for this snapshot. Proceed with repo_commit.",
        }, ensure_ascii=False, indent=2)

    # Run the advisory review
    advisory_backend = "Claude Code" if has_anthropic_advisory else "openai-compatible"
    ctx.emit_progress_fn(f"Running advisory pre-review ({advisory_backend}, read-only)...")
    changed_files = _get_changed_file_list(repo_dir, paths=paths)

    # Fail closed if git status itself is broken — proceeding with a broken file list
    # would let advisory review proceed on incomplete context.
    if changed_files.startswith("⚠️ ADVISORY_ERROR"):
        return json.dumps({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": changed_files,
            "message": (
                "Advisory review aborted: could not retrieve changed file list. "
                "Fix the error and retry, or use skip_advisory_pre_review=True to bypass (will be audited)."
            ),
        }, ensure_ascii=False, indent=2)

    # Cheap deterministic version-sync check before expensive SDK call.
    # Reads worktree content (advisory runs before git add, so no staged index).
    # Non-fatal: a warning note is prepended to the advisory prompt context but
    # does not abort the advisory run.
    version_sync_warning = _check_worktree_version_sync(repo_dir)
    if version_sync_warning:
        ctx.emit_progress_fn(f"⚠️ Advisory preflight: {version_sync_warning}")

    if has_anthropic_advisory:
        items, raw_result = _run_claude_advisory(
            repo_dir,
            commit_message,
            ctx,
            goal=goal,
            scope=scope,
            paths=paths,
            drive_root=drive_root,
        )
    else:
        items, raw_result = _run_llm_advisory(
            repo_dir,
            commit_message,
            ctx,
            goal=goal,
            scope=scope,
            paths=paths,
            drive_root=drive_root,
        )

    # Handle errors from the CLI
    if raw_result.startswith("⚠️ ADVISORY_ERROR"):
        return json.dumps({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": raw_result,
            "message": (
                "Advisory review failed to run. Fix the error and retry, "
                "or use skip_advisory_pre_review=True to bypass (will be audited)."
            ),
        }, ensure_ascii=False, indent=2)

    # Budget gate: prompt too large — non-blocking skip (mirrors scope review).
    # Persist a durable "skipped" run so _check_advisory_freshness treats this
    # snapshot as having been reviewed (is_fresh returns True for status="skipped").
    if raw_result.startswith("⚠️ ADVISORY_SKIPPED:"):
        snapshot_summary = f"{changed_files.count(chr(10)) + 1} file(s) changed"
        def _mutate_skip(skip_state: AdvisoryReviewState) -> None:
            next_run_attempt = len(
                skip_state.filter_advisory_runs(
                    repo_key=repo_key,
                    tool_name="advisory_pre_review",
                    task_id=task_id,
                )
            ) + 1
            skip_state.add_run(AdvisoryRunRecord(
                snapshot_hash=snapshot_hash,
                commit_message=commit_message,
                status="skipped",
                ts=_utc_now(),
                items=[],
                snapshot_summary=snapshot_summary,
                raw_result=raw_result,
                snapshot_paths=paths,
                repo_key=repo_key,
                tool_name="advisory_pre_review",
                task_id=task_id,
                attempt=next_run_attempt,
            ))

        update_state(drive_root, _mutate_skip)
        return json.dumps({
            "status": "skipped",
            "snapshot_hash": snapshot_hash,
            "message": raw_result,
        }, ensure_ascii=False, indent=2)

    # Classify findings
    critical_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() == "critical"]
    advisory_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() != "critical"]

    snapshot_summary = f"{changed_files.count(chr(10)) + 1} file(s) changed"

    # If items is empty but raw_result is non-empty, the advisory ran but failed to parse.
    # Treat this as a parse_failure to avoid silently treating it as an all-clear.
    run_status = "fresh" if items else "parse_failure"
    run = AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status=run_status,
        ts=_utc_now(),
        items=items,
        snapshot_summary=snapshot_summary,
        raw_result=raw_result,
        snapshot_paths=paths,
        repo_key=repo_key,
        tool_name="advisory_pre_review",
        task_id=task_id,
        attempt=len(
            state.filter_advisory_runs(
                repo_key=repo_key,
                tool_name="advisory_pre_review",
                task_id=task_id,
            )
        ) + 1,
    )
    state.add_run(run)

    # Surface parse failures as explicit errors (not silent all-clears)
    if run_status == "parse_failure":
        save_state(drive_root, state)
        return json.dumps({
            "status": "parse_failure",
            "snapshot_hash": snapshot_hash,
            "error": "Advisory ran but returned no parseable checklist items.",
            "raw_result": _truncate_review_artifact(raw_result),
            "message": (
                "Advisory output could not be parsed. Re-run advisory_pre_review, "
                "or use skip_advisory_pre_review=True to bypass (will be audited)."
            ),
        }, ensure_ascii=False, indent=2)

    # Always try to resolve open obligations from parseable advisory results.
    # _resolve_matching_obligations only resolves when PASS is unambiguous (no concurrent FAIL
    # for the same item), so it is safe to call even when critical_fails is non-empty.
    # An obligation whose checklist item now passes should be resolved regardless of whether
    # *other* unrelated items still fail — leaving it open would turn unrelated criticals into
    # a perpetual hard gate on closed obligations.
    if items:
        _resolve_matching_obligations(state, items, snapshot_hash, repo_key=repo_key)

    save_state(drive_root, state)

    # Build human-readable summary
    findings_summary: List[str] = []
    for item in critical_fails:
        findings_summary.append(f"  CRITICAL [{item.get('item','?')}]: {item.get('reason','')}")
    for item in advisory_fails:
        findings_summary.append(f"  ADVISORY [{item.get('item','?')}]: {item.get('reason','')}")

    result = {
        "status": "fresh",
        "snapshot_hash": snapshot_hash,
        "ts": run.ts,
        "items": items,
        "critical_count": len(critical_fails),
        "advisory_count": len(advisory_fails),
        "snapshot_summary": snapshot_summary,
        "message": (
            f"Advisory review complete. {len(critical_fails)} critical, "
            f"{len(advisory_fails)} advisory findings. "
            "Fix issues and run repo_commit when ready."
        ),
    }
    if findings_summary:
        result["findings"] = findings_summary

    return json.dumps(result, ensure_ascii=False, indent=2)


def _handle_review_status(
    ctx: ToolContext,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: Optional[int] = None,
) -> str:
    """Show recent advisory pre-review run history AND last commit attempt state.

    Includes: advisory run history, staleness from edits, open obligations from
    blocking rounds, and a concrete next-step recommendation.
    """
    drive_root = pathlib.Path(ctx.drive_root)
    state = load_state(drive_root)
    repo_dir_value = getattr(ctx, "repo_dir", "")
    repo_dir = (
        pathlib.Path(repo_dir_value)
        if isinstance(repo_dir_value, (str, pathlib.Path)) and str(repo_dir_value)
        else None
    )
    repo_filter = repo_key or (make_repo_key(repo_dir) if repo_dir is not None else None)
    tool_filter = tool_name or None
    task_filter = task_id or None
    filtered_runs = state.filter_advisory_runs(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    filtered_attempts = state.filter_attempts(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )

    runs_data = []
    for run in reversed(filtered_runs):  # full history — no [-5:] cap
        findings = [i for i in (run.items or []) if isinstance(i, dict)
                    and str(i.get("verdict", "")).upper() == "FAIL"]
        critical = [i for i in findings if str(i.get("severity", "")).lower() == "critical"]
        runs_data.append({
            "snapshot_hash": run.snapshot_hash[:12],
            "commit_message": run.commit_message,  # full — no [:80] truncation
            "status": run.status,
            "ts": run.ts,  # full ts — no [:16] truncation
            "critical_findings": len(critical),
            "total_findings": len(findings),
            "snapshot_summary": run.snapshot_summary,
            "bypass_reason": run.bypass_reason or None,
            "repo_key": run.repo_key or None,
            "tool_name": run.tool_name or None,
            "task_id": run.task_id or None,
            "attempt": int(run.attempt or 0) or None,
        })

    latest = filtered_runs[-1] if filtered_runs else None

    # Compute current snapshot hash using the same paths scope as the latest run
    # so path-scoped advisories don't appear falsely stale.
    try:
        if repo_dir is None:
            raise ValueError("repo_dir unavailable")
        latest_paths = latest.snapshot_paths if latest else None
        current_hash = compute_snapshot_hash(repo_dir, "", paths=latest_paths)
        hash_mismatch = bool(
            latest and latest.status in ("fresh", "bypassed", "skipped", "parse_failure")
            and latest.snapshot_hash != current_hash
        )
    except Exception:
        current_hash = None
        hash_mismatch = False

    # Gate-accurate freshness: look up the run matching the CURRENT hash,
    # not just `latest` — handles restored snapshots where an older fresh run exists.
    open_obs = state.get_open_obligations(repo_key=repo_filter)
    matching_run = state.find_by_hash(current_hash, repo_key=repo_filter) if current_hash else None
    effective_is_fresh = bool(
        state.is_fresh(current_hash, repo_key=repo_filter) if current_hash else False
    )
    # Use matching_run for guidance; fall back to latest for history display
    guidance_run = matching_run or latest

    # Staleness: either explicit edit-invalidation OR live hash mismatch
    stale_from_edit = bool(
        hash_mismatch or (
            state.last_stale_from_edit_ts
            and state.last_stale_repo_key in ("", repo_filter)
        )
    )
    stale_from_edit_ts = (
        state.last_stale_from_edit_ts  # full ts — no [:16] truncation
        if state.last_stale_from_edit_ts and state.last_stale_repo_key in ("", repo_filter)
        else ("now (hash mismatch)" if hash_mismatch else None)
    )
    stale_reason = (
        state.last_stale_reason
        if state.last_stale_repo_key in ("", repo_filter)
        else ""
    ) or (
        "Current snapshot hash no longer matches the latest advisory run."
        if hash_mismatch else None
    )

    # Build commit attempt section
    selected_attempt = filtered_attempts[-1] if filtered_attempts else (
        None if (repo_filter or tool_filter or task_filter or attempt is not None) else state.last_commit_attempt
    )
    commit_attempt_data = None
    if selected_attempt:
        ca = selected_attempt
        commit_attempt_data = {
            "status": ca.status,
            "commit_message": ca.commit_message,  # full — no [:80] truncation
            "ts": ca.ts,  # full ts — no [:16] truncation
            "duration_sec": round(ca.duration_sec, 1),
            "block_reason": ca.block_reason or None,
            "block_details_preview": _truncate_review_artifact(ca.block_details, limit=300) if ca.block_details else None,
            "repo_key": ca.repo_key or None,
            "tool_name": ca.tool_name or None,
            "task_id": ca.task_id or None,
            "attempt": int(ca.attempt or 0) or None,
            "phase": ca.phase or None,
            "blocked": bool(ca.blocked),
            "late_result_pending": bool(ca.late_result_pending),
            "critical_findings": len(ca.critical_findings or []),
            "advisory_findings": len(ca.advisory_findings or []),
            "obligation_ids": list(ca.obligation_ids or []),
            "readiness_warnings": list(ca.readiness_warnings or []),
            "pre_review_fingerprint": ca.pre_review_fingerprint[:12] or None,
            "post_review_fingerprint": ca.post_review_fingerprint[:12] or None,
            "fingerprint_status": ca.fingerprint_status or None,
            "degraded_reasons": list(ca.degraded_reasons or []),
        }

    attempts_data = []
    for entry in reversed(filtered_attempts):  # full history — no [-8:] cap
        attempts_data.append({
            "repo_key": entry.repo_key or None,
            "tool_name": entry.tool_name or None,
            "task_id": entry.task_id or None,
            "attempt": int(entry.attempt or 0) or None,
            "phase": entry.phase or None,
            "status": entry.status,
            "blocked": bool(entry.blocked),
            "late_result_pending": bool(entry.late_result_pending),
            "critical_findings": len(entry.critical_findings or []),
            "advisory_findings": len(entry.advisory_findings or []),
            "obligation_ids": list(entry.obligation_ids or []),
            "readiness_warnings": list(entry.readiness_warnings or []),
            "pre_review_fingerprint": entry.pre_review_fingerprint[:12] or None,
            "post_review_fingerprint": entry.post_review_fingerprint[:12] or None,
            "fingerprint_status": entry.fingerprint_status or None,
            "degraded_reasons": list(entry.degraded_reasons or []),
            "ts": entry.ts,  # full ts — no [:16] truncation
        })

    # Open obligations (already computed above as open_obs for effective_is_fresh)
    obligations_data = []
    for ob in open_obs:
        obligations_data.append({
            "obligation_id": ob.obligation_id,
            "item": ob.item,
            "severity": ob.severity,
            "reason": _truncate_review_artifact(ob.reason, limit=200),
            "status": ob.status,
            "source_ts": ob.source_attempt_ts,  # full ts — no [:16] truncation
            "source_commit": ob.source_attempt_msg,  # full message — no [:60] truncation
        })

    # Determine readiness and actionable next step (via module-level helper)

    # Build human-readable summary
    ca = selected_attempt
    if ca and ca.status in ("blocked", "failed"):
        reason_map = {
            "no_advisory": "No fresh advisory pre-review found. Run advisory_pre_review first.",
            "critical_findings": "Reviewers found critical issues. Fix all issues listed, then re-run advisory.",
            "review_quorum": "Not enough review models responded. Retry — usually transient.",
            "parse_failure": "Review models could not produce parseable output. Retry the commit.",
            "infra_failure": "Infrastructure failure (git lock, git command, or review API). Check block_details.",
            "scope_blocked": "Scope reviewer blocked the commit. Address scope review findings.",
            "preflight": "Preflight check failed (missing VERSION/README). Stage all related files.",
            "revalidation_failed": "The staged diff changed after review. Re-run advisory and review on the final staged diff.",
            "fingerprint_unavailable": "The staged diff could not be fingerprinted for revalidation. Fix the git diff issue and retry.",
            "overlap_guard": "Another reviewed attempt is still active. Wait for it to finish or expire before retrying.",
        }
        block_action = reason_map.get(
            ca.block_reason,
            f"{ca.status}: {ca.block_reason or 'unknown'}. Check block_details."
        )
        label = "BLOCKED" if ca.status == "blocked" else "FAILED"
        if ca.late_result_pending:
            block_action += " Late result is still pending."
        status_msg = f"Last commit {label} ({ca.block_reason or 'unclassified'}): {block_action}"
    else:
        # Use effective (gate-accurate) status derived from live snapshot, not latest run
        pass  # status_msg set below after effective_status is computed

    next_step_msg = _next_step_guidance(guidance_run, state, stale_from_edit, stale_from_edit_ts,
                                         open_obs, effective_is_fresh=effective_is_fresh)
    # Derive primary status fields from current snapshot (gate-accurate), not from latest run.
    # matching_run is the advisory that matches the live worktree hash (may differ from latest).
    # If a matching run exists for this hash, use its actual status (including "parse_failure")
    # rather than collapsing to "stale" — that would hide the real gate-relevant state.
    # Only fall back to "stale"/"none" when there is NO matching run for the current snapshot.
    if matching_run:
        effective_status = matching_run.status
    elif latest:
        effective_status = "stale"
    else:
        effective_status = "none"
    effective_hash = (
        matching_run.snapshot_hash[:12] if matching_run and matching_run.snapshot_hash else None
    )
    # status_summary / message MUST be derived from the effective (gate-accurate) state only.
    # Never show "latest run status" here — that can be from a different snapshot and is
    # confusing / internally contradictory.  If a blocking commit attempt is recorded we
    # prepend that context, but the current-snapshot status always closes the sentence.
    if ca and ca.status in ("blocked", "failed"):
        # Keep the block-action sentence from above but append the live advisory state so
        # they remain consistent even when the worktree has changed since the last block.
        status_msg = f"{status_msg}  |  Current advisory: {effective_status}"
    else:
        status_msg = f"Current advisory: {effective_status}"

    return json.dumps({
        "latest_advisory_status": effective_status,
        "latest_advisory_hash": effective_hash,
        "stale_from_edit": stale_from_edit,
        "stale_from_edit_ts": stale_from_edit_ts,
        "stale_reason": stale_reason,
        "filters": {
            "repo_key": repo_filter,
            "tool_name": tool_filter,
            "task_id": task_filter,
            "attempt": attempt,
        },
        "advisory_runs": runs_data,
        "attempts": attempts_data,
        "last_commit_attempt": commit_attempt_data,
        "open_obligations": obligations_data,
        "open_obligations_count": len(obligations_data),
        "status_summary": status_msg,
        "message": status_msg,  # backward-compat alias for status_summary
        "next_step": next_step_msg,
    }, ensure_ascii=False, indent=2)


# -- Tool registration --

def get_tools() -> list:
    return [
        ToolEntry(
            name="advisory_pre_review",
            schema={
                "name": "advisory_pre_review",
                "description": (
                    "Run an advisory pre-commit review via Claude Agent SDK (read-only: Read, Grep, Glob only). "
                    "MUST be called before repo_commit. Returns structured JSON findings. "
                    "Findings are advisory (non-blocking), but the absence of a fresh matching "
                    "advisory run will block repo_commit. "
                    "Correct workflow: finish edits -> advisory_pre_review(...) -> repo_commit(...) immediately. "
                    "WARNING: any edit (repo_write/str_replace_editor) after advisory_pre_review "
                    "automatically marks advisory as stale and requires re-running it. "
                    "Use skip_advisory_pre_review=True to bypass (bypass is durably audited)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commit_message": {
                            "type": "string",
                            "description": "Intended commit message. Used to bind the advisory run to this specific commit.",
                        },
                        "skip_advisory_pre_review": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Explicitly bypass the advisory review. "
                                "Bypass is durably audited in events.jsonl. "
                                "Default: False."
                            ),
                        },
                        "goal": {
                            "type": "string",
                            "description": "High-level goal of this change. Used to judge completeness.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "Declared scope boundary. Issues outside scope are advisory-only.",
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Explicit list of changed file paths. Auto-detected from git status if omitted.",
                        },
                    },
                    "required": ["commit_message"],
                },
            },
            handler=_handle_advisory_pre_review,
        ),
        ToolEntry(
            name="review_status",
            schema={
                "name": "review_status",
                "description": (
                    "Show recent advisory pre-review run history. "
                    "Read-only diagnostic — use to check if a fresh advisory run exists "
                    "before calling repo_commit. Also shows: last commit attempt state "
                    "(reviewing/blocked/succeeded/failed) with block reason and actionable guidance; "
                    "whether advisory is stale because of a worktree edit; "
                    "open obligations from previous blocking rounds; "
                    "and a concrete next_step recommendation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_key": {
                            "type": "string",
                            "description": "Optional repo identity filter for attempt/advisory history.",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Optional tool-name filter (for example repo_commit or repo_write_commit).",
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Optional task-id filter for attempt/advisory history.",
                        },
                        "attempt": {
                            "type": "integer",
                            "description": "Optional attempt number filter within the selected repo/tool/task scope.",
                        },
                    },
                    "required": [],
                },
            },
            handler=_handle_review_status,
        ),
    ]
