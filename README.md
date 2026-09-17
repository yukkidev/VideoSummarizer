# VideoSummarizer

Watch any video, get a clean summary, then **ask it anything** — all local:
no API bills, no data leaving your machine.

```
download (yt-dlp) → transcribe (faster-whisper) → summary + Q&A (Ollama)
```

Three interfaces share the same engine:

- **Web GUI** — video player, clickable timestamp citations, built-in downloader,
  model switcher with live refresh
- **TUI** — fast terminal interface for development
- **CLI** — scriptable, JSON output for piping

## Requirements

- Python 3.10+
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) on `PATH` (or installed in the venv)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (yt-dlp uses it for audio extraction)
- [Ollama](https://ollama.com/) running locally with at least one model
- Optional: `faster-whisper` for audio transcription; without it, the app falls
  back to video subtitles

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[whisper,media,dev]"
```

Extras: `whisper` (faster-whisper), `media` (yt-dlp), `dev` (pytest, ruff).

## Quick start

```bash
# CLI: download + transcribe + summarize
.venv/bin/vidsum run "https://www.youtube.com/watch?v=..."

# Ask a question (one-shot), or omit the question for a threaded chat REPL
.venv/bin/vidsum ask <video_id_or_dir> "What did they say about X?"
.venv/bin/vidsum ask <video_id_or_dir> --new --mode chat   # new conversation
.venv/bin/vidsum ask --global "What do my videos say about focus?"
.venv/bin/vidsum threads <video_id_or_dir>                  # list conversations

# List processed videos / switch models
.venv/bin/vidsum list
.venv/bin/vidsum models --refresh
.venv/bin/vidsum models --set qwen3:8b

# Terminal UI
.venv/bin/vidsum tui

# Web GUI (opens http://127.0.0.1:8765/)
.venv/bin/vidsum gui
```

The legacy entry point still works: `python -m pipeline.runner <url>`.

## Web GUI

- Paste a link on the left: choose **video / audio / subtitles**, force a redo,
  or use **subtitles only** to skip Whisper.
- A live progress bar tracks the background job, and a **browser notification**
  fires when it finishes (the server also sends a `notify-send` desktop
  notification, so you can walk away and close the tab).
- The player streams from disk with HTTP Range support (seeking works).
- Every timestamp in summaries, answers, and the transcript is **clickable** and
  jumps the player to that moment. Q&A answers also show scored source chips.
- The **Ask** tab is a persistent conversation: pick a thread, start a **New**
  one, delete one, and toggle **Ask** (strictly grounded, cited) vs **Chat**
  (conversational, history-aware). Follow-ups remember the whole conversation.
- The **Context** button shows how full the window is using Ollama's own
  numbers: the context actually loaded for the model (`/api/ps`), the model's
  maximum (`/api/show`), the last request's prompt/generated token counts, and
  a conversation estimate.
- **Playlists** in the sidebar group related videos (create with ＋, expand to
  add/remove videos, delete without touching the library).
- **All videos · global chat** talks to the library as a whole. Its **Scope**
  picker is a checkbox dropdown of playlists: select one or more to restrict
  retrieval to just those videos, so unrelated topics never crowd the context.
  Citations name the video plus timestamp (click one to jump to that video at
  that moment).
- The top bar lists installed models, marks the one **loaded in memory**, and a
  refresh button re-queries Ollama at any time. Switching models updates
  `data/config.json` immediately and is picked up by every interface.

## Models

- Default: `ornith-1.5:9b` (configurable)
- Switch anywhere: `vidsum models --set <model>`, the GUI dropdown, or the TUI
  `m` overlay
- Env overrides: `VS_MODEL`, `VS_WHISPER_MODEL`, `VS_OLLAMA_URL`, `VS_DATA_DIR`
- Pull new models from the GUI (`Pull…`) or with `ollama pull <model>`

## Transcription

1. If `faster-whisper` is installed and audio exists, audio is transcribed
   (default model `small`; override with `--whisper-model` or `VS_WHISPER_MODEL`).
2. Otherwise, downloaded subtitles (manual or automatic) are parsed.
3. `--subtitles` / "subtitles only" forces the subtitle path.

Whisper models download from Hugging Face on first use and are cached by
`faster-whisper`. Use `tiny`/`base` for speed, `small`+ for accuracy.

## Q&A and conversations

Questions are answered by retrieval over transcript chunks, with full
conversation history so follow-ups build on each other:

1. Chunks are embedded locally with `nomic-embed-text` via Ollama during
   processing (cached per video in `data/<video_id>/vectors.json`).
2. Top-k chunks are retrieved by cosine similarity.
3. Two modes share the retrieval:
   - **ask** — strictly grounded in the excerpts; every claim cites `[mm:ss]`
   - **chat** — conversational and history-aware; can connect ideas and use
     general knowledge, still citing the video when it draws on it.

Conversations are stored as threads in `data/chats/`, one JSON file each. Every
video has its own threads, and **global** threads retrieve across the whole
library (answers cite `title @ [mm:ss]`). A global thread can be **scoped to
playlists**: only videos in the selected playlists are searched, and the exact
scope is passed to the model as part of the prompt. Legacy `answers.jsonl`
histories are migrated into an "Earlier questions" thread on first open.

`GET /api/search?q=...&playlists=<id,id>` exposes the same scoped semantic
search directly — the `api.search_library` facade an MCP server or external
tool could wrap.

## Agent layer

```bash
.venv/bin/vidsum agents                    # scripted safety demo
.venv/bin/vidsum agents --agent analyst --task "Summarize the latest video"
.venv/bin/vidsum agents --agent researcher --url "https://..." --interactive
```

- Agents have **charters**: allowed tools, allowed write paths, and protected
  core files (`agents/framework.py`, `pipeline/models.py`, ...).
- Agents can call `list_videos`, `get_video`, `ask_video`, and the new
  `search_library` (cross-video semantic recall).
- Every tool has a danger level: `safe`, `guarded`, or `dangerous`.
- `HumanGate` is the single choke point: destructive actions are **denied by
  default**, prompt in `--interactive` mode, and only run hands-off with
  `--auto` (explicit).
- Every decision is appended to `data/agents/audit.jsonl`.
- `Agent.run()` is a real Ollama tool loop with tolerant JSON parsing, repeated
  call detection, and error feedback.

## Data layout

```
data/
  config.json               active model + settings (incl. desktop_notify)
  chats/<thread_id>.json    conversation threads (video + global, with scope)
  playlists/<id>.json       playlists and their video ids
  jNQXAC9IVRw/
    metadata.json           title, author, duration, paths
    *.mp4 *.mp3 *.vtt       downloaded media
    transcript.json         text + timestamped segments
    chunks.json             retrieval/summary chunks
    summary.json / .md      summary + questions
    vectors.json            embedding cache
    answers.jsonl           legacy Q&A log (auto-migrated to a thread)
  agents/audit.jsonl        gate decisions
```

## Testing

```bash
.venv/bin/python -m pytest -m "not live"   # fast suite, no network
.venv/bin/python -m pytest -m live         # real Ollama + whisper + network
.venv/bin/ruff check .
```

## Troubleshooting

- **"No video formats found" / 403s from YouTube** — update yt-dlp:
  `.venv/bin/pip install -U yt-dlp` (the venv copy is preferred automatically).
- **"faster-whisper is not installed"** — `pip install -e ".[whisper]"`, or use
  `--subtitles`.
- **"cannot reach Ollama"** — make sure `ollama serve` is running and a model is
  pulled; `vidsum models --refresh` shows the live state.
- **Long videos** are summarized with map-reduce (per-section notes, then a
  final synthesis), so context limits are not a problem.
