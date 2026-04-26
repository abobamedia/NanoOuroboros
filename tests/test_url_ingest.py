"""Tests for ouroboros.tools.url_ingest."""

from __future__ import annotations

import json
import pathlib
from typing import List

import pytest

from ouroboros.tools import url_ingest
from ouroboros.tools.registry import ToolContext


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the workspace boundary at a temp directory for the test."""
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
    return tmp_path


def _ctx(tmp_path: pathlib.Path) -> ToolContext:
    return ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")


def test_drive_match_recognizes_folder():
    kind, ident = url_ingest._drive_match(
        "https://drive.google.com/drive/folders/1dHEEkkGYLQtlTTxM9iY2_AOFLV9XQnp1?usp=sharing"
    )
    assert kind == "folder"
    assert ident == "1dHEEkkGYLQtlTTxM9iY2_AOFLV9XQnp1"


def test_drive_match_recognizes_file_path_form():
    kind, ident = url_ingest._drive_match(
        "https://drive.google.com/file/d/10USod9ojo7vSkzQYiPUOyubP1kvyU_ma/view?usp=sharing"
    )
    assert kind == "file"
    assert ident == "10USod9ojo7vSkzQYiPUOyubP1kvyU_ma"


def test_drive_match_recognizes_uc_form():
    kind, ident = url_ingest._drive_match(
        "https://drive.google.com/uc?export=download&id=1fOx__r2nPnu11De7mQDBNtWGsYqOPRZX"
    )
    assert kind == "file"
    assert ident == "1fOx__r2nPnu11De7mQDBNtWGsYqOPRZX"


def test_drive_match_rejects_non_drive():
    kind, ident = url_ingest._drive_match("https://example.com/file.csv")
    assert kind == ""
    assert ident == ""


def test_sanitize_filename_strips_traversal():
    assert url_ingest._sanitize_filename("../../etc/passwd") == "passwd"
    assert url_ingest._sanitize_filename("/abs/path/data.csv") == "data.csv"
    assert url_ingest._sanitize_filename("clean name.csv") == "clean name.csv"
    assert url_ingest._sanitize_filename("") == "download.bin"
    assert url_ingest._sanitize_filename("..") == "download.bin"


def test_resolve_dest_dir_inside_workspace(workspace):
    target = url_ingest._resolve_dest_dir("ingested/yandex")
    assert target.exists()
    assert workspace in target.parents or target == workspace / "ingested" / "yandex"


def test_resolve_dest_dir_rejects_escape(workspace, tmp_path):
    outside = tmp_path.parent / "outside-workspace"
    with pytest.raises(ValueError, match="escapes workspace"):
        url_ingest._resolve_dest_dir(str(outside))


def test_drive_file_uses_gdown(workspace, monkeypatch, tmp_path):
    captured: dict = {}

    def fake_download(*, id, output, quiet, fuzzy):
        captured.update({"id": id, "output": output, "fuzzy": fuzzy})
        out_dir = pathlib.Path(output.rstrip("/"))
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = out_dir / "report.csv"
        produced.write_text("h1,h2\n1,2\n", encoding="utf-8")
        return str(produced)

    monkeypatch.setattr(
        "gdown.download", fake_download, raising=False,
    )
    result = url_ingest._read_url_impl(
        _ctx(tmp_path),
        url="https://drive.google.com/file/d/abc1234567890/view",
        dest_dir="ingested/x",
    )
    assert result["ok"] is True
    assert captured["id"] == "abc1234567890"
    assert len(result["files"]) == 1
    assert result["files"][0]["name"] == "report.csv"
    assert result["bytes_total"] > 0


def test_drive_folder_collects_new_files(workspace, monkeypatch, tmp_path):
    def fake_download_folder(*, id, output, quiet, use_cookies):
        out_dir = pathlib.Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "a.csv").write_text("a", encoding="utf-8")
        (out_dir / "b.csv").write_text("bb", encoding="utf-8")

    monkeypatch.setattr(
        "gdown.download_folder", fake_download_folder, raising=False,
    )
    result = url_ingest._read_url_impl(
        _ctx(tmp_path),
        url="https://drive.google.com/drive/folders/folderid12345",
        dest_dir="ingested/folder",
    )
    assert result["ok"] is True
    names = sorted(f["name"] for f in result["files"])
    assert names == ["a.csv", "b.csv"]
    assert result["bytes_total"] == 3


def test_http_url_uses_urllib(workspace, monkeypatch, tmp_path):
    class FakeResp:
        headers = {"Content-Disposition": 'attachment; filename="data.csv"'}
        def __init__(self) -> None:
            self._chunks = [b"alpha,beta\n", b"1,2\n"]
        def read(self, _n: int) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""
        def __enter__(self) -> "FakeResp":
            return self
        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(_req, timeout: int):
        return FakeResp()

    monkeypatch.setattr(url_ingest.urllib.request, "urlopen", fake_urlopen)
    result = url_ingest._read_url_impl(
        _ctx(tmp_path),
        url="https://example.com/path/data.csv",
        dest_dir="ingested/http",
    )
    assert result["ok"] is True
    assert len(result["files"]) == 1
    assert result["files"][0]["name"] == "data.csv"
    assert result["files"][0]["bytes"] > 0


def test_http_url_enforces_max_bytes(workspace, monkeypatch, tmp_path):
    class HugeResp:
        headers: dict = {}
        def __init__(self) -> None:
            self._sent = 0
        def read(self, n: int) -> bytes:
            if self._sent > 200_000:
                return b""
            self._sent += n
            return b"x" * n
        def __enter__(self) -> "HugeResp":
            return self
        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        url_ingest.urllib.request, "urlopen",
        lambda _req, timeout: HugeResp(),
    )
    result = url_ingest._read_url_impl(
        _ctx(tmp_path),
        url="https://example.com/huge.bin",
        max_bytes=64 * 1024,
    )
    assert result["ok"] is False
    assert "max_bytes" in (result.get("error") or "")
    # No partial file should be left behind.
    leftovers = list((workspace / "ingested").rglob("*"))
    assert all(not p.is_file() or p.stat().st_size == 0 for p in leftovers)


def test_handler_returns_json(workspace, monkeypatch, tmp_path):
    def fake_download(*, id, output, quiet, fuzzy):
        out_dir = pathlib.Path(output.rstrip("/"))
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = out_dir / "x.csv"
        produced.write_text("x", encoding="utf-8")
        return str(produced)

    monkeypatch.setattr("gdown.download", fake_download, raising=False)
    raw = url_ingest._read_url(
        _ctx(tmp_path),
        url="https://drive.google.com/file/d/idABCDEFGHIJ/view",
    )
    decoded = json.loads(raw)
    assert decoded["ok"] is True
    assert decoded["files"][0]["name"] == "x.csv"


def test_tool_entry_registered():
    entries = url_ingest.get_tools()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "read_url"
    assert "url" in entry.schema["parameters"]["properties"]
    assert entry.schema["parameters"]["required"] == ["url"]


def test_url_ingest_in_frozen_module_list():
    from ouroboros.tools.registry import ToolRegistry
    assert "url_ingest" in ToolRegistry._FROZEN_TOOL_MODULES
