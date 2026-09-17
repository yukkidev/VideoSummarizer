"""Curses terminal UI for VideoSummarizer.

Keys
----
q          quit                 p     process a new URL
j/k ↓/↑    navigate             a     ask / keep talking
Tab        switch pane          n     new conversation
1/2/3      summary/questions/   t     cycle conversations
           transcript           M     toggle ask/chat mode
                                m     model switcher
                                o     open media at selected timestamp
                                r     reload video list
"""

from __future__ import annotations

import curses
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

from pipeline import api
from pipeline.llm import LLMError
from pipeline.models import format_timestamp

TABS = ("summary", "questions", "transcript")


def wrap_lines(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text]
    lines: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        current = ""
        for word in raw.split():
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def format_video_line(record: dict[str, Any]) -> str:
    title = (record.get("title") or record.get("video_id") or "?")[:60]
    return title


def format_citation_lines(answer: dict[str, Any], width: int) -> list[str]:
    lines = []
    for citation in answer.get("citations") or []:
        stamp = format_timestamp(citation.get("start", 0))
        score = citation.get("score", 0)
        text = (citation.get("text") or "").replace("\n", " ")[: width - 20]
        lines.append(f"  [{stamp}] ({score:.2f}) {text}")
    return lines


def player_command(path: str, seconds: float = 0.0) -> list[str] | None:
    seconds = max(0.0, float(seconds or 0))
    for name, cmd in (
        ("mpv", ["mpv", f"--start={seconds:.0f}", "--force-window=yes"]),
        ("ffplay", ["ffplay", "-ss", f"{seconds:.0f}", "-autoexit"]),
        ("vlc", ["vlc", f"--start-time={seconds:.0f}"]),
    ):
        if shutil.which(name):
            return cmd + [path]
    if shutil.which("xdg-open"):
        return ["xdg-open", path]
    return None


def open_media(path: str, seconds: float = 0.0) -> bool:
    command = player_command(path, seconds)
    if not command or not path:
        return False
    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


@dataclass
class TuiState:
    out_dir: str = "data"
    videos: list[dict[str, Any]] = field(default_factory=list)
    selected: int = 0
    tab: str = "summary"
    scroll: int = 0
    detail: dict[str, Any] | None = None
    threads: list[dict[str, Any]] = field(default_factory=list)
    thread_id: str = ""
    thread: dict[str, Any] | None = None
    thread_mode: str = "ask"
    status: str = "ready"
    busy: bool = False
    models: dict[str, Any] | None = None
    pane: str = "list"
    config_model: str = ""

    @property
    def current_video_id(self) -> str | None:
        if not self.videos:
            return None
        return self.videos[self.selected].get("video_id")

    def clamp(self) -> None:
        if self.videos:
            self.selected = max(0, min(self.selected, len(self.videos) - 1))
        else:
            self.selected = 0

    def content_lines(self, width: int) -> list[str]:
        detail = self.detail or {}
        if self.tab == "summary":
            summary = detail.get("summary") or {}
            lines = wrap_lines(summary.get("summary") or "(no summary)", width)
            if summary.get("model"):
                lines.insert(0, f"model: {summary['model']}")
            return lines
        if self.tab == "questions":
            summary = detail.get("summary") or {}
            lines = []
            for index, question in enumerate(summary.get("highlights") or [], 1):
                lines.extend(wrap_lines(f"{index}. {question}", width))
                lines.append("")
            return lines or ["(no questions)"]
        chunks = detail.get("chunks") or []
        lines = []
        for chunk in chunks:
            stamp = format_timestamp(chunk.get("start", 0))
            first = True
            for wrapped in wrap_lines(chunk.get("text") or "", width - 12):
                lines.append(f"[{stamp}] {wrapped}" if first else f"         {wrapped}")
                first = False
            lines.append("")
        return lines or ["(no transcript)"]

    def qa_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        messages = (self.thread or {}).get("messages") or []
        for message in messages:
            role = message.get("role")
            prefix = "you> " if role == "user" else "bot> "
            lines.extend(wrap_lines(prefix + (message.get("content") or ""), width))
            if role != "user":
                lines.extend(format_citation_lines(message, width))
            lines.append("")
        return lines or ["(no messages in this conversation)"]

    def selected_timestamp(self) -> float:
        detail = self.detail or {}
        if self.tab != "transcript":
            return 0.0
        chunks = detail.get("chunks") or []
        if not chunks:
            return 0.0
        index = min(self.scroll // 3, len(chunks) - 1)
        return float(chunks[max(0, index)].get("start") or 0.0)


def load_state(state: TuiState) -> None:
    state.videos = api.list_videos(out_dir=state.out_dir)
    state.clamp()
    state.detail = None
    state.threads = []
    state.thread_id = ""
    state.thread = None
    selected = state.current_video_id
    if selected:
        load_detail(state, selected)


def load_threads(state: TuiState, video_id: str, keep: str = "") -> None:
    try:
        state.threads = api.list_threads(video_id, out_dir=state.out_dir)
    except LLMError as exc:
        state.status = str(exc)
        state.threads = []
    wanted = keep if any(t.get("thread_id") == keep for t in state.threads) else ""
    if not wanted and state.threads:
        wanted = state.threads[0].get("thread_id") or ""
    state.thread_id = wanted
    state.thread = None
    if wanted:
        try:
            state.thread = api.get_thread(wanted, out_dir=state.out_dir)
            state.thread_mode = state.thread.get("mode") or state.thread_mode
        except LLMError as exc:
            state.status = str(exc)
            state.thread = None


def load_detail(state: TuiState, video_id: str) -> None:
    keep = state.thread_id
    try:
        state.detail = api.get_video_record(video_id, out_dir=state.out_dir)
    except LLMError as exc:
        state.status = str(exc)
        state.detail = None
        state.threads = []
        state.thread = None
        state.thread_id = ""
        return
    load_threads(state, video_id, keep=keep)


def new_thread(state: TuiState) -> None:
    video_id = state.current_video_id
    if not video_id:
        state.status = "no video selected"
        return
    try:
        thread = api.create_thread(
            video_id, mode=state.thread_mode, out_dir=state.out_dir,
        )
    except LLMError as exc:
        state.status = str(exc)
        return
    state.threads.insert(0, {
        "thread_id": thread.get("thread_id", ""),
        "title": thread.get("title", ""),
        "mode": thread.get("mode", "ask"),
        "message_count": 0,
    })
    state.thread_id = thread.get("thread_id", "")
    state.thread = thread
    state.status = "new conversation"


def cycle_thread(state: TuiState) -> None:
    ids = [t.get("thread_id") for t in state.threads]
    if not ids:
        state.status = "no conversations yet"
        return
    try:
        index = ids.index(state.thread_id)
    except ValueError:
        index = -1
    next_id = ids[(index + 1) % len(ids)]
    state.thread_id = next_id or ""
    try:
        state.thread = api.get_thread(next_id, out_dir=state.out_dir)
        state.thread_mode = state.thread.get("mode") or state.thread_mode
        state.status = f"thread: {state.thread.get('title') or next_id}"
    except LLMError as exc:
        state.status = str(exc)


class Worker:
    def __init__(self, state: TuiState) -> None:
        self.state = state
        self.updates: queue.Queue[tuple[str, str]] = queue.Queue()

    def _run(self, label: str, fn) -> None:
        self.state.busy = True
        self.updates.put(("status", label))

        def progress(message: str, fraction: float) -> None:
            self.updates.put(("status", f"{message} ({int(fraction * 100)}%)"))

        def target() -> None:
            try:
                fn(progress)
            except Exception as exc:  # noqa: BLE001 - shown in the status bar
                self.updates.put(("error", str(exc)))
            finally:
                self.updates.put(("done", ""))

        threading.Thread(target=target, daemon=True).start()

    def refresh_models(self) -> None:
        def task(progress) -> None:
            self.state.models = api.models_status(config=api.load_config(self.state.out_dir))
            self.state.config_model = self.state.models.get("current", self.state.config_model)

        self._run("refreshing models", task)

    def switch_model(self, model: str) -> None:
        def task(progress) -> None:
            api.set_model(model, data_dir=self.state.out_dir)
            self.state.models = api.models_status(model, config=api.load_config(self.state.out_dir))
            self.state.config_model = model

        self._run(f"switching to {model}", task)

    def process(self, url: str) -> None:
        def task(progress) -> None:
            record = api.process(url, out_dir=self.state.out_dir, progress=progress)
            self.updates.put(("open", record.get("video_id", "")))

        self._run(f"processing {url}", task)

    def ask(self, question: str) -> None:
        video_id = self.state.current_video_id
        if not video_id:
            return

        def task(progress) -> None:
            thread_id = self.state.thread_id
            if not thread_id:
                thread = api.create_thread(
                    video_id, mode=self.state.thread_mode, out_dir=self.state.out_dir,
                )
                thread_id = thread.get("thread_id", "")
                self.updates.put(("thread", thread))
            api.post_message(
                thread_id, question, mode=self.state.thread_mode,
                out_dir=self.state.out_dir, progress=progress,
            )

        self._run("thinking", task)

    def drain(self) -> None:
        while True:
            try:
                kind, payload = self.updates.get_nowait()
            except queue.Empty:
                return
            if kind == "status":
                self.state.status = payload
            elif kind == "error":
                self.state.status = f"error: {payload}"
                self.state.busy = False
            elif kind == "done":
                self.state.busy = False
                if self.state.current_video_id:
                    load_detail(self.state, self.state.current_video_id)
            elif kind == "thread":
                if payload and payload.get("thread_id"):
                    self.state.thread_id = payload["thread_id"]
                    self.state.thread = payload
            elif kind == "open" and payload:
                self.state.videos = api.list_videos(out_dir=self.state.out_dir)
                self.state.clamp()
                for index, video in enumerate(self.state.videos):
                    if video.get("video_id") == payload:
                        self.state.selected = index
                load_detail(self.state, payload)


def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _draw(stdscr, state: TuiState) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 10 or width < 50:
        _safe_addstr(stdscr, 0, 0, "terminal too small", curses.A_BOLD)
        stdscr.refresh()
        return

    model_line = f"model: {state.config_model}"
    if state.models:
        loaded = "loaded" if state.models.get("current_loaded") else "not loaded"
        server = "online" if state.models.get("server_ok") else "OFFLINE"
        model_line = f"model: {state.models.get('current')} [{loaded}] | ollama {server}"
    _safe_addstr(stdscr, 0, 1, " VideoSummarizer ", curses.A_BOLD | curses.A_REVERSE)
    _safe_addstr(stdscr, 0, width - len(model_line) - 2, model_line, curses.A_DIM)
    _safe_addstr(stdscr, 1, 1, "─" * (width - 2), curses.A_DIM)

    list_width = min(42, max(24, width // 3))
    _safe_addstr(stdscr, 2, 1, "Library", curses.A_BOLD)
    if not state.videos:
        _safe_addstr(stdscr, 3, 1, "(empty - press p to add)", curses.A_DIM)
    for index, video in enumerate(state.videos[: height - 6]):
        marker = ">" if index == state.selected else " "
        attr = curses.A_REVERSE if index == state.selected and state.pane == "list" else 0
        _safe_addstr(
            stdscr, 3 + index, 1,
            f"{marker} {format_video_line(video)[:list_width - 4]:<{list_width - 4}}",
            attr,
        )

    content_x = list_width + 2
    content_width = width - content_x - 2
    header = " | ".join(
        (f"[{tab}]" if tab == state.tab else tab) for tab in TABS
    ) + " | q&a"
    if state.thread:
        thread_title = (state.thread.get("title") or "new conversation")[:24]
        header += f"  [{state.thread_mode}] {thread_title}"
    _safe_addstr(stdscr, 2, content_x, header, curses.A_BOLD)
    if state.detail:
        title = (state.detail.get("title") or "")[:content_width]
        _safe_addstr(stdscr, 3, content_x, title, curses.A_BOLD)
        offset = 4
    else:
        offset = 3

    if state.tab == "transcript":
        lines = state.content_lines(content_width)
    else:
        lines = state.content_lines(content_width)
    visible = height - offset - 4
    scroll = max(0, min(state.scroll, max(0, len(lines) - visible)))
    for row, line in enumerate(lines[scroll:scroll + visible]):
        _safe_addstr(stdscr, offset + row, content_x, line[:content_width])

    qa_start = height - 3
    _safe_addstr(stdscr, qa_start - 1, 1, "─" * (width - 2), curses.A_DIM)
    qa_line = state.qa_lines(content_width)
    if qa_line:
        _safe_addstr(stdscr, height - 2, 1, qa_line[-1][: width - 2], curses.A_DIM)

    status = state.status
    if state.busy:
        status = "⏳ " + status
    _safe_addstr(stdscr, height - 1, 1, status[: width - 2], curses.A_REVERSE)
    keys = " q:quit p:add a:ask n:new t:thread M:mode m:model o:open 1/2/3:tabs r:reload "
    _safe_addstr(stdscr, height - 2, max(1, width - len(keys) - 2), keys, curses.A_DIM)
    stdscr.refresh()


def _prompt(stdscr, label: str) -> str:
    height, width = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    try:
        _safe_addstr(stdscr, height - 1, 1, " " * (width - 2))
        _safe_addstr(stdscr, height - 1, 1, label, curses.A_BOLD)
        raw = stdscr.getstr(height - 1, 1 + len(label), width - len(label) - 3)
        return raw.decode(errors="replace").strip()
    finally:
        curses.noecho()
        curses.curs_set(0)


def _model_switcher(stdscr, state: TuiState, worker: Worker) -> None:
    if state.models is None:
        worker.refresh_models()
        return
    models = [m["name"] for m in state.models.get("installed", [])]
    if not models:
        state.status = "no models installed"
        return
    height, width = stdscr.getmaxyx()
    box_height = min(len(models) + 4, height - 4)
    box_width = min(60, width - 4)
    top = (height - box_height) // 2
    left = (width - box_width) // 2
    win = curses.newwin(box_height, box_width, top, left)
    win.box()
    win.addstr(0, 2, " select model ", curses.A_BOLD)
    index = 0
    current = state.models.get("current")
    if current in models:
        index = models.index(current)
    while True:
        win.addstr(1, 2, "Enter=use  r=refresh  Esc=cancel", curses.A_DIM)
        for row, name in enumerate(models[: box_height - 3]):
            marker = ">" if row == index else " "
            attr = curses.A_REVERSE if row == index else 0
            if name == current:
                attr |= curses.A_BOLD
            win.addstr(2 + row, 2, f"{marker} {name:<40}"[: box_width - 4], attr)
        win.refresh()
        key = win.getch()
        if key in (curses.KEY_UP, ord("k")):
            index = max(0, index - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            index = min(len(models) - 1, index + 1)
        elif key in (10, 13):
            worker.switch_model(models[index])
            return
        elif key == ord("r"):
            worker.refresh_models()
            return
        elif key in (27, ord("q")):
            return


def run_tui(out_dir: str = "data") -> None:
    state = TuiState(out_dir=out_dir)
    worker = Worker(state)
    try:
        cfg = api.load_config(out_dir)
        cfg.save()
        state.config_model = cfg.model
    except OSError:
        pass
    load_state(state)
    worker.refresh_models()

    def main(stdscr) -> None:
        curses.curs_set(0)
        stdscr.timeout(150)
        curses.use_default_colors()
        while True:
            worker.drain()
            _draw(stdscr, state)
            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord("q"),):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                if state.pane == "list":
                    state.selected += 1
                    state.clamp()
                    load_detail(state, state.current_video_id or "")
                else:
                    state.scroll += 1
            elif key in (curses.KEY_UP, ord("k")):
                if state.pane == "list":
                    state.selected -= 1
                    state.clamp()
                    load_detail(state, state.current_video_id or "")
                else:
                    state.scroll = max(0, state.scroll - 1)
            elif key == ord("\t"):
                state.pane = "content" if state.pane == "list" else "list"
            elif key in (ord("1"),):
                state.tab, state.scroll = "summary", 0
            elif key in (ord("2"),):
                state.tab, state.scroll = "questions", 0
            elif key in (ord("3"),):
                state.tab, state.scroll = "transcript", 0
            elif key in (curses.KEY_NPAGE,):
                state.scroll += 10
            elif key in (curses.KEY_PPAGE,):
                state.scroll = max(0, state.scroll - 10)
            elif key == ord("r"):
                load_state(state)
                state.status = "reloaded"
            elif key == ord("R"):
                worker.refresh_models()
            elif key == ord("m"):
                _model_switcher(stdscr, state, worker)
            elif key == ord("p"):
                url = _prompt(stdscr, "url> ")
                if url and not worker.state.busy:
                    worker.process(url)
            elif key in (ord("a"),):
                question = _prompt(stdscr, "ask> ")
                if not question:
                    continue
                if state.busy:
                    state.status = "busy - wait for the current task"
                else:
                    worker.ask(question)
            elif key in (ord("n"),):
                if state.busy:
                    state.status = "busy - wait for the current task"
                else:
                    new_thread(state)
            elif key in (ord("t"),):
                cycle_thread(state)
            elif key in (ord("M"),):
                state.thread_mode = "chat" if state.thread_mode == "ask" else "ask"
                state.status = f"mode: {state.thread_mode}"
            elif key == ord("o"):
                detail = state.detail or {}
                path = detail.get("video_path") or detail.get("audio_path")
                if path:
                    opened = open_media(path, state.selected_timestamp())
                    state.status = "opened player" if opened else "no player found (mpv/ffplay/vlc)"
                else:
                    state.status = "no media downloaded"

    curses.wrapper(main)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VideoSummarizer TUI")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    run_tui(out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
