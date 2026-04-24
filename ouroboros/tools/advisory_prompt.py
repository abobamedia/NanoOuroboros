"""Prompt construction for advisory pre-review."""

from __future__ import annotations

import pathlib
import subprocess
from typing import List, Optional

from ouroboros.review_state import load_state, make_repo_key
from ouroboros.tools.review_helpers import (
    CRITICAL_FINDING_CALIBRATION,
    build_blocking_findings_json_section,
    build_goal_section,
    build_scope_section,
    load_checklist_section,
)


def _load_doc(repo_dir: pathlib.Path, relpath: str, fallback: str = "") -> str:
    try:
        p = repo_dir / relpath
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return fallback


def _build_blocking_history_section(drive_root: pathlib.Path, repo_key: str = "") -> str:
    try:
        state = load_state(drive_root)
    except Exception:
        return ""
    return build_blocking_findings_json_section(
        state.get_open_obligations(repo_key=repo_key),
        state.get_blocking_history(repo_key=repo_key),
    )


def build_advisory_prompt(
    repo_dir: pathlib.Path,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    resolved_paths: Optional[List[str]] = None,
    drive_root: Optional[pathlib.Path] = None,
    diff: Optional[str] = None,
    changed_files: Optional[str] = None,
    touched_pack: str = "",
    omitted_paths: Optional[List[str]] = None,
) -> str:
    """Build the read-only advisory review prompt."""
    bible = _load_doc(repo_dir, "BIBLE.md", "(BIBLE.md not found)")
    try:
        checklists = load_checklist_section("Repo Commit Checklist")
    except Exception:
        checklists = _load_doc(repo_dir, "docs/CHECKLISTS.md", "(CHECKLISTS.md not found)")
    dev_guide = _load_doc(repo_dir, "docs/DEVELOPMENT.md", "(DEVELOPMENT.md not found)")
    arch_doc = _load_doc(repo_dir, "docs/ARCHITECTURE.md", "(ARCHITECTURE.md not found)")
    if diff is None:
        try:
            path_args = (["--"] + list(resolved_paths)) if resolved_paths else []
            staged = subprocess.run(
                ["git", "diff", "--cached"] + path_args,
                cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
            )
            unstaged = subprocess.run(
                ["git", "diff"] + path_args,
                cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
            )
            diff = ((staged.stdout or "") + (unstaged.stdout or "")).strip() or "(no unstaged/staged changes found)"
        except Exception as exc:
            diff = f"⚠️ ADVISORY_ERROR: failed to retrieve diff: {exc}"
    if changed_files is None:
        try:
            path_args = (["--"] + list(resolved_paths)) if resolved_paths else []
            result = subprocess.run(
                ["git", "status", "--porcelain"] + path_args,
                cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
            )
            lines = [line.rstrip() for line in (result.stdout or "").splitlines() if line.strip()]
            changed_files = "\n".join(lines) if lines else "(clean — no changed files)"
        except Exception as exc:
            changed_files = f"⚠️ ADVISORY_ERROR: git status error: {exc}"
    goal_section = build_goal_section(goal, scope, commit_message)
    scope_section = build_scope_section(scope)
    blocking_history = _build_blocking_history_section(drive_root, make_repo_key(repo_dir)) if drive_root else ""

    omitted_note = ""
    if omitted_paths:
        preview = ", ".join(list(omitted_paths)[:5])
        if len(omitted_paths) > 5:
            preview += f", +{len(omitted_paths) - 5} more"
        omitted_note = f"\n*(Inline pack contains omission notes for {len(omitted_paths)} path(s): {preview})*\n"

    critical_calibration = CRITICAL_FINDING_CALIBRATION

    return f"""\
You are performing a pre-commit review of an Ouroboros self-modifying AI agent codebase.

## Your role — NON-NEGOTIABLE REQUIREMENTS
- Review the current working tree changes with the SAME RIGOR as the downstream blocking reviewers.
  A false PASS here wastes an entire blocking review cycle ($10+).
- Use ONLY Read, Grep, Glob tools. Do NOT edit or execute any files.
- Read the FULL CONTENT of every changed file listed below using the Read tool.
  Do NOT evaluate security, bible compliance, or code quality from path listings or diff hunks alone.
- Return ONLY a JSON array. No prose, no markdown fences — only the JSON array.

## Thoroughness requirements
- Do NOT stop after finding the first issue. Check EVERY item in the checklist.
- Report ALL problems you find. If there are 5 bugs, list all 5 — each as a separate entry.
- Do NOT summarize multiple distinct problems into one finding.
- For PASS: brief reason is fine. For FAIL: cite the specific file, line/symbol, what is wrong,
  and provide a CONCRETE fix suggestion so the developer knows exactly what to change.

## Severity thresholds — treat as blocking reviewers do
- bible_compliance (item 1): ANY violation of BIBLE.md principles is CRITICAL.
- security_issues (item 5): ANY path traversal, secret leakage, or unsafe operation is CRITICAL.
- development_compliance (item 2): naming, entity type rules, module size, no ad-hoc LLM calls,
  no hardcoded [:N] truncation of cognitive artifacts — all CRITICAL when violated.
- self_consistency (item 13): if a concrete stale artifact exists (specific file + line), CRITICAL.

## Critical finding calibration (shared with triad and scope reviewers)

{critical_calibration}

## Output format
Return ONLY a JSON array. Each element:
{{
  "item": "<checklist item name>",
  "verdict": "PASS" | "FAIL",
  "severity": "critical" | "advisory",
  "reason": "<for FAIL: file, line/symbol, what is wrong, how to fix>"
}}

## CHECKLISTS.md (What to review)

{checklists}

{scope_section}

{goal_section}

## DEVELOPMENT.md (Engineering standards)

{dev_guide}

## BIBLE.md (Constitutional context — top priority)

{bible}

## ARCHITECTURE.md (System structure — critical for version sync and module checks)

{arch_doc}

{blocking_history}

## Commit message

{commit_message}

## Changed files (git status --porcelain)

{changed_files}

## Current touched files (full content — read these with the Read tool for deeper inspection)

{touched_pack}
{omitted_note}

## Staged diff

{diff}

## Step-by-step instructions
1. Read the FULL content of every changed file using the Read tool. Do not skip any file.
2. Check EVERY item from the "Repo Commit Checklist" — do not stop after the first issue.
3. Pay equal attention to ALL 13 checklist items. bible_compliance and security_issues must be
   evaluated at the same strictness as the downstream blocking reviewers.
4. Look for ALL bugs, logic errors, regressions, race conditions, and violations of BIBLE.md or DEVELOPMENT.md.
5. Cross-check: do tool descriptions in prompts match actual get_tools() exports?
   Does ARCHITECTURE.md header version match the VERSION file?
6. **MANDATORY — Prior obligations:** If an "Unresolved obligations" section appears above,
   address EVERY listed obligation explicitly in your output:
   a. Include a separate JSON entry per obligation for the corresponding checklist item.
   b. If fixed: verdict=PASS, reason must state WHAT closes it (file, line, symbol, change).
   c. If not fixed: verdict=FAIL, severity=critical, reason must name the specific stale artifact.
   d. **TARGETING — multiple obligations with the same checklist item:**
      When two or more open obligations share the same item (e.g. two distinct `code_quality`
      findings), you MUST emit a separate JSON entry for EACH one and use the
      `(obligation <id>)` suffix in the `"item"` field to target it precisely:
        {{"item": "code_quality (obligation abc123def456)", "verdict": "PASS", ...}}
      A generic `"item": "code_quality"` entry when multiple same-item obligations are
      open will NOT resolve all of them — only the one matched by `obligation_id` will
      be closed; the rest remain open until explicitly addressed.
7. Output ONLY the JSON array — no markdown fences, no commentary outside the JSON.
"""
