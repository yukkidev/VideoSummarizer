"""Tools exposed to agents, wired to the video pipeline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from agents.framework import DANGEROUS, GUARDED, SAFE, Tool
from pipeline import api

MAX_READ_BYTES = 200_000
COMMAND_ALLOWLIST = {
    "pytest", "python", "python3", "ls", "cat", "rg", "grep", "wc",
    "head", "tail", "ffprobe", "ffmpeg", "yt-dlp",
}


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"not found: {path}"
    if p.stat().st_size > MAX_READ_BYTES:
        return f"file too large to read (> {MAX_READ_BYTES} bytes): {path}"
    try:
        return p.read_text(errors="replace")
    except OSError as exc:
        return f"error reading {path}: {exc}"


def _list_dir(path: str = ".") -> str:
    p = Path(path)
    if not p.is_dir():
        return f"not a directory: {path}"
    entries = sorted(os.listdir(p))[:200]
    return "\n".join(entries)


def _search_web(query: str) -> str:
    try:
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "20",
             f"https://duckduckgo.com/html/?q={query}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"search failed: {exc}"
    return proc.stdout[:4000] or "no results"


def _summarize_video(url: str, out_dir: str = "data") -> dict[str, Any]:
    record = api.process(url, out_dir=out_dir, want_video=False)
    return {
        "video_id": record.get("video_id"),
        "title": record.get("title"),
        "summary": (record.get("summary") or {}).get("summary", ""),
        "questions": (record.get("summary") or {}).get("highlights", []),
    }


def _ask_video(video_id: str, question: str, out_dir: str = "data") -> dict[str, Any]:
    payload = api.ask(video_id, question, out_dir=out_dir)
    return payload.get("answer", {})


def _write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def _edit_code(path: str, old: str, new: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"not found: {path}"
    data = p.read_text()
    if old not in data:
        return "old string not found; no edit made"
    p.write_text(data.replace(old, new, 1))
    return f"edited {path}"


def _run_command(command: str, timeout: int = 120) -> str:
    first = (command.strip().split() or [""])[0]
    base = Path(first).name
    if base not in COMMAND_ALLOWLIST:
        return (
            f"command '{base}' is not allowlisted; allowed: "
            f"{', '.join(sorted(COMMAND_ALLOWLIST))}"
        )
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out after {timeout}s"
    output = (proc.stdout or "") + (proc.stderr or "")
    if len(output) > 6000:
        output = output[:6000] + "\n…[truncated]"
    return f"exit={proc.returncode}\n{output}"


def build_tools(*, out_dir: str = "data") -> dict[str, Tool]:
    def summarize(url: str) -> dict[str, Any]:
        return _summarize_video(url, out_dir)

    def ask(video_id: str, question: str) -> dict[str, Any]:
        return _ask_video(video_id, question, out_dir)

    def list_videos() -> list[dict[str, Any]]:
        return api.list_videos(out_dir=out_dir)

    def get_video(video_id: str) -> dict[str, Any]:
        return api.get_video_record(video_id, out_dir=out_dir)

    def search_library(query: str) -> list[dict[str, Any]]:
        return api.search_library(query, out_dir=out_dir)["citations"]

    tools = [
        Tool(
            name="list_videos",
            description="List processed videos with their flags and ids.",
            func=list_videos,
            danger=SAFE,
        ),
        Tool(
            name="get_video",
            description="Get summary, transcript chunks, and metadata for a video id.",
            func=get_video,
            danger=SAFE,
        ),
        Tool(
            name="search_library",
            description=(
                "Search transcript moments across every processed video by meaning; "
                "hits cite video title and timestamp."
            ),
            func=search_library,
            danger=SAFE,
            params={"query": "string"},
        ),
        Tool(
            name="ask_video",
            description="Ask a question about a processed video; answers cite timestamps.",
            func=ask,
            danger=SAFE,
            params={"video_id": "string", "question": "string"},
        ),
        Tool(
            name="read_file",
            description="Read a text file.",
            func=_read_file,
            danger=SAFE,
            params={"path": "string"},
        ),
        Tool(
            name="list_dir",
            description="List directory entries.",
            func=_list_dir,
            danger=SAFE,
            params={"path": "string"},
        ),
        Tool(
            name="search_web",
            description="Search the web (network access).",
            func=_search_web,
            danger=GUARDED,
            params={"query": "string"},
        ),
        Tool(
            name="summarize_video",
            description="Download, transcribe, and summarize a video URL (network + disk).",
            func=summarize,
            danger=DANGEROUS,
            params={"url": "string"},
        ),
        Tool(
            name="write_file",
            description="Write a text file (charter-restricted).",
            func=_write_file,
            danger=GUARDED,
            path_arg="path",
            params={"path": "string", "content": "string"},
        ),
        Tool(
            name="edit_code",
            description="Replace one occurrence of a string in a file (charter-restricted).",
            func=_edit_code,
            danger=GUARDED,
            path_arg="path",
            params={"path": "string", "old": "string", "new": "string"},
        ),
        Tool(
            name="run_command",
            description="Run an allowlisted shell command.",
            func=_run_command,
            danger=GUARDED,
            params={"command": "string"},
        ),
        Tool(
            name="download_model",
            description="Pull an Ollama model (network + disk).",
            func=lambda model: __import__("subprocess").run(
                ["ollama", "pull", model], capture_output=True, text=True, timeout=3600,
            ).stdout[-2000:],
            danger=DANGEROUS,
            params={"model": "string"},
        ),
    ]
    return {tool.name: tool for tool in tools}
