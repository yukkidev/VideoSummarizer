from __future__ import annotations

from gui.tui import (
    TuiState,
    format_citation_lines,
    format_video_line,
    player_command,
    wrap_lines,
)


def test_wrap_lines_basic():
    assert wrap_lines("one two three", 7) == ["one two", "three"]
    assert wrap_lines("", 10) == [""]
    assert wrap_lines("a\nb", 10) == ["a", "b"]
    assert wrap_lines("supercalifragilistic", 5) == ["supercalifragilistic"]


def test_format_video_line_truncates():
    line = format_video_line({"title": "x" * 100})
    assert len(line) == 60
    assert format_video_line({"video_id": "abc"}) == "abc"


def test_format_citation_lines():
    lines = format_citation_lines(
        {"citations": [{"start": 65.0, "score": 0.9, "text": "hello world"}]},
        width=60,
    )
    assert len(lines) == 1
    assert "[01:05]" in lines[0]
    assert "0.90" in lines[0]


def test_player_command_prefers_mpv(monkeypatch):
    monkeypatch.setattr("gui.tui.shutil.which", lambda name: "/usr/bin/mpv" if name == "mpv" else None)
    assert player_command("/v.mp4", 12.0) == ["mpv", "--start=12", "--force-window=yes", "/v.mp4"]


def test_player_command_ffplay_fallback(monkeypatch):
    monkeypatch.setattr("gui.tui.shutil.which", lambda name: "/usr/bin/ffplay" if name == "ffplay" else None)
    assert player_command("/v.mp4", 12.0) == ["ffplay", "-ss", "12", "-autoexit", "/v.mp4"]


def test_player_command_xdg_fallback(monkeypatch):
    monkeypatch.setattr("gui.tui.shutil.which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None)
    assert player_command("/v.mp4", 0) == ["xdg-open", "/v.mp4"]


def test_player_command_none(monkeypatch):
    monkeypatch.setattr("gui.tui.shutil.which", lambda name: None)
    assert player_command("/v.mp4", 0) is None


def make_state() -> TuiState:
    state = TuiState()
    state.videos = [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}]
    return state


def test_selection_clamps():
    state = make_state()
    state.selected = 99
    state.clamp()
    assert state.selected == 2
    state.selected = -5
    state.clamp()
    assert state.selected == 0
    state.videos = []
    state.selected = 3
    state.clamp()
    assert state.selected == 0
    assert state.current_video_id is None


def test_content_lines_summary():
    state = TuiState()
    state.detail = {"summary": {"summary": "hello world", "model": "m"}}
    lines = state.content_lines(40)
    assert "model: m" in lines[0]
    assert "hello world" in lines[1]


def test_content_lines_questions():
    state = TuiState()
    state.tab = "questions"
    state.detail = {"summary": {"highlights": ["why?", "how?"]}}
    lines = state.content_lines(40)
    assert any("1. why?" in line for line in lines)
    assert any("2. how?" in line for line in lines)


def test_content_lines_transcript_includes_stamps():
    state = TuiState()
    state.tab = "transcript"
    state.detail = {"chunks": [{"start": 65.0, "text": "hello"}]}
    lines = state.content_lines(40)
    assert any(line.startswith("[01:05]") for line in lines)


def test_content_lines_empty_placeholders():
    state = TuiState()
    assert state.content_lines(40) == ["(no summary)"]
    state.tab = "questions"
    assert state.content_lines(40) == ["(no questions)"]
    state.tab = "transcript"
    assert state.content_lines(40) == ["(no transcript)"]


def test_qa_lines_with_citations():
    state = TuiState()
    state.thread = {
        "title": "t",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "citations": [{"start": 5, "score": 0.5, "text": "t"}],
            },
        ],
    }
    lines = state.qa_lines(50)
    assert any(line.startswith("you>") for line in lines)
    assert any(line.startswith("bot>") for line in lines)
    assert any("[00:05]" in line for line in lines)
    assert TuiState().qa_lines(50) == ["(no messages in this conversation)"]


def test_selected_timestamp():
    state = TuiState()
    state.tab = "transcript"
    state.detail = {"chunks": [{"start": 0.0}, {"start": 12.0}, {"start": 30.0}]}
    assert state.selected_timestamp() == 0.0
    state.scroll = 3
    assert state.selected_timestamp() == 12.0
    state.scroll = 300
    assert state.selected_timestamp() == 30.0
    state.tab = "summary"
    assert state.selected_timestamp() == 0.0
