"""URL ingestion tool — pull files from public URLs (incl. Google Drive) into the workspace.

Built so Ouroboros can answer "read this link" requests in chat — for example
when Vlad pastes a Google Drive folder of Yandex Direct exports and asks for
analysis. The tool downloads to a sandboxed destination inside the workspace
and returns metadata that downstream skills (direct_ingestion, etc.) can pick
up by path.

Sandbox-safety: destination must live inside the workspace root
(`~/AI/ouroboros-workspace/` by default, overridable by env
`OUROBOROS_FILE_BROWSER_DEFAULT`). Filenames are sanitized; absolute paths
and traversal segments are rejected.

Drive support is implemented via `gdown` (already in the venv). Plain HTTP
falls back to `urllib.request` with a hard byte cap so a malicious URL
cannot fill the disk.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import pathlib
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


# 500 MB hard cap per single URL fetch — enough for very large exports while
# protecting the workspace volume from a runaway link.
_DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_DEFAULT_TIMEOUT_SEC = 120

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\- ]")


def _workspace_root() -> pathlib.Path:
    """The sandbox boundary for ingested files."""
    raw = os.environ.get("OUROBOROS_FILE_BROWSER_DEFAULT", "").strip()
    if raw:
        return pathlib.Path(raw).expanduser().resolve()
    return (pathlib.Path.home() / "AI" / "ouroboros-workspace").resolve()


def _resolve_dest_dir(dest_dir: Optional[str]) -> pathlib.Path:
    """Resolve and validate the destination directory; must stay inside workspace."""
    root = _workspace_root()
    if dest_dir:
        candidate = pathlib.Path(dest_dir).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate)
    else:
        candidate = root / "ingested"
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"dest_dir {candidate!s} escapes workspace boundary {root!s}"
        ) from exc
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _sanitize_filename(name: str, fallback: str = "download.bin") -> str:
    """Strip path separators and unsafe characters from a filename."""
    base = pathlib.PurePosixPath(str(name or "")).name.strip()
    if not base or base in (".", ".."):
        return fallback
    cleaned = _FILENAME_SAFE_RE.sub("_", base)
    return cleaned or fallback


def _drive_match(url: str) -> Tuple[str, str]:
    """Identify a Google Drive URL.

    Returns (kind, id) where kind is "folder", "file", or "" when not Drive.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return ("", "")
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host and "docs.google.com" not in host:
        return ("", "")
    path = parsed.path or ""
    folder_match = re.search(r"/folders/([A-Za-z0-9_-]{10,})", path)
    if folder_match:
        return ("folder", folder_match.group(1))
    file_match = re.search(r"/file/d/([A-Za-z0-9_-]{10,})", path)
    if file_match:
        return ("file", file_match.group(1))
    qs = urllib.parse.parse_qs(parsed.query or "")
    if qs.get("id"):
        return ("file", qs["id"][0])
    return ("", "")


def _list_recursive(root: pathlib.Path) -> List[pathlib.Path]:
    """List all regular files under root, sorted by path for deterministic output."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def _file_meta(path: pathlib.Path, base_dir: pathlib.Path) -> Dict[str, Any]:
    """Report metadata for a single ingested file."""
    try:
        rel = str(path.relative_to(base_dir))
    except ValueError:
        rel = path.name
    mime, _ = mimetypes.guess_type(path.name)
    return {
        "path": str(path),
        "rel": rel,
        "name": path.name,
        "bytes": path.stat().st_size,
        "mime_type_guess": mime or "application/octet-stream",
    }


def _download_drive_file(file_id: str, dest_dir: pathlib.Path) -> List[pathlib.Path]:
    """Download a single Drive file, returning the resulting paths."""
    import gdown  # local import — heavy dep, keep tool import-fast

    before = set(_list_recursive(dest_dir))
    output = gdown.download(
        id=file_id,
        output=str(dest_dir) + "/",
        quiet=True,
        fuzzy=True,
    )
    if output:
        # gdown returns a path string; honour it directly.
        produced = pathlib.Path(output).resolve()
        if produced.is_file():
            return [produced]
    after = set(_list_recursive(dest_dir))
    return sorted(after - before)


def _download_drive_folder(folder_id: str, dest_dir: pathlib.Path) -> List[pathlib.Path]:
    """Download a public Drive folder; returns list of files written."""
    import gdown

    before = set(_list_recursive(dest_dir))
    gdown.download_folder(
        id=folder_id,
        output=str(dest_dir),
        quiet=True,
        use_cookies=False,
    )
    after = set(_list_recursive(dest_dir))
    return sorted(after - before)


def _download_http(
    url: str,
    dest_dir: pathlib.Path,
    *,
    max_bytes: int,
    timeout_sec: int,
) -> List[pathlib.Path]:
    """Download a plain HTTP(s) URL with a hard byte cap."""
    parsed = urllib.parse.urlparse(url)
    fallback_name = pathlib.PurePosixPath(parsed.path or "").name or "download.bin"
    req = urllib.request.Request(url, headers={"User-Agent": "Ouroboros-url-ingest/1"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        # Prefer Content-Disposition filename when available.
        disp = resp.headers.get("Content-Disposition") or ""
        match = re.search(r'filename\*=UTF-8\'\'([^;]+)|filename="?([^";]+)"?', disp)
        if match:
            disp_name = urllib.parse.unquote(match.group(1) or match.group(2) or "")
            fallback_name = disp_name or fallback_name
        out_name = _sanitize_filename(fallback_name)
        out_path = dest_dir / out_name
        bytes_read = 0
        with out_path.open("wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    fh.close()
                    out_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"download exceeded max_bytes={max_bytes}; aborted at {bytes_read}"
                    )
                fh.write(chunk)
    return [out_path]


def _read_url_impl(
    ctx: ToolContext,
    url: str,
    dest_dir: Optional[str] = None,
    max_bytes: Optional[int] = None,
    timeout_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Concrete logic — separated so tests can call it without JSON-encoding."""
    cleaned_url = (url or "").strip()
    if not cleaned_url:
        return {"ok": False, "error": "url is empty", "files": [], "bytes_total": 0,
                "source_url": "", "notes": ""}

    cap = int(max_bytes) if max_bytes is not None else _DEFAULT_MAX_BYTES
    timeout = int(timeout_sec) if timeout_sec is not None else _DEFAULT_TIMEOUT_SEC

    try:
        target_dir = _resolve_dest_dir(dest_dir)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "files": [], "bytes_total": 0,
                "source_url": cleaned_url, "notes": ""}

    kind, ident = _drive_match(cleaned_url)
    notes_parts: List[str] = []
    try:
        if kind == "folder":
            written = _download_drive_folder(ident, target_dir)
            notes_parts.append("drive folder via gdown")
        elif kind == "file":
            written = _download_drive_file(ident, target_dir)
            notes_parts.append("drive file via gdown (handles >100MB confirm)")
        else:
            written = _download_http(
                cleaned_url, target_dir, max_bytes=cap, timeout_sec=timeout,
            )
            notes_parts.append("plain http via urllib")
    except Exception as exc:
        log.warning("read_url failed for %s: %s", cleaned_url, exc, exc_info=True)
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "files": [],
            "bytes_total": 0,
            "source_url": cleaned_url,
            "notes": "; ".join(notes_parts),
        }

    files_meta = [_file_meta(p, target_dir) for p in written if p.is_file()]
    return {
        "ok": True,
        "files": files_meta,
        "bytes_total": sum(f["bytes"] for f in files_meta),
        "source_url": cleaned_url,
        "dest_dir": str(target_dir),
        "notes": "; ".join(notes_parts),
        "error": None,
    }


def _read_url(ctx: ToolContext, **kwargs: Any) -> str:
    """JSON-string handler for the agent loop."""
    result = _read_url_impl(
        ctx,
        url=kwargs.get("url", ""),
        dest_dir=kwargs.get("dest_dir"),
        max_bytes=kwargs.get("max_bytes"),
        timeout_sec=kwargs.get("timeout_sec"),
    )
    return json.dumps(result, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "read_url",
            {
                "name": "read_url",
                "description": (
                    "Download a public URL into the sandboxed workspace and return file "
                    "metadata. Handles Google Drive file/folder links (including >100MB "
                    "files that need a confirm-token via gdown) and plain HTTP(S). "
                    "Destination defaults to ~/AI/ouroboros-workspace/ingested/ and is "
                    "always sandboxed inside the workspace root. Use this when the user "
                    "shares a link (Drive export, dataset, etc.) and asks you to ingest "
                    "or analyse it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Public URL to ingest (Drive folder/file or http).",
                        },
                        "dest_dir": {
                            "type": "string",
                            "description": (
                                "Sub-directory under workspace (relative or absolute). "
                                "Must resolve inside the workspace boundary."
                            ),
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": "Hard cap for HTTP downloads (default 500 MB).",
                        },
                        "timeout_sec": {
                            "type": "integer",
                            "description": "Per-request timeout in seconds (default 120).",
                        },
                    },
                    "required": ["url"],
                },
            },
            _read_url,
            timeout_sec=900,
        ),
    ]
