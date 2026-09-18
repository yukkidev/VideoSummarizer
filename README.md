# VideoSummarizer

Watch any video, get a summary, then **ask it anything**. All running locally using local tools and local models.

```
download (yt-dlp) → transcribe (faster-whisper) → summary + Q&A (Ollama)
```

Three interfaces share the same engine:

- **Web GUI** — player, clickable timestamp citations, built-in downloader,
  model switcher with live refresh
- **TUI** — fast terminal interface for development
- **CLI** — scriptable, JSON output for piping

## Requirements

Install everything below. The app has fallbacks, but you only get the full
feature set — audio transcription, semantic search, grounded Q&A — when all of
these are present.

### 1. ffmpeg

Required by yt-dlp to extract audio and merge video.

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` |

### 2. Ollama

Runs every model locally. Install, start it, and pull **both** models:

| Platform | Install |
|----------|---------|
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS | `brew install ollama` or [ollama.com/download](https://ollama.com/download) |
| Windows | [ollama.com/download](https://ollama.com/download) |

```bash
ollama serve                   # skip if Ollama already runs as a service
ollama pull ornith-1.5:9b      # default chat model (summaries + Q&A)
ollama pull nomic-embed-text   # required for semantic search / retrieval
```

`nomic-embed-text` is **not optional**: every video is embedded during
processing so Q&A and global chat can search it. Pull both before your first
run.

> **Raise Ollama's context window.** It defaults to a small 4096 tokens. On
> Linux, set `Environment="OLLAMA_CONTEXT_LENGTH=8192"` in the Ollama service
> (`sudo systemctl edit ollama.service`), then
> `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

On low-end or low-VRAM hardware, switch to a lighter chat model such as
`qwen3:4b`, `qwen3:8b`, or `gemma4:e2b`. The plain `gemma4:e2b` tag resolves to
the 7.2 GB q4_K_M quant; for the smaller 4.3 GB QAT build, pull the explicit
tag:

```bash
ollama pull gemma4:e2b-it-qat
vidsum models --set gemma4:e2b-it-qat
```

### 3. faster-whisper and yt-dlp

Both are Python packages and ship as extras in the install step below. Do not
skip them:

- **faster-whisper** transcribes the audio. Without it, the app falls back to
  downloaded subtitles (and fails when a video has none).
- **yt-dlp** downloads video, audio, and subtitles.

### Optional extras

- **Desktop notifications** when a job finishes (Linux): `notify-send` from
  libnotify (`sudo apt install libnotify-bin` or `sudo dnf install libnotify`).
  Browser notifications work without it.
- **TUI "open media"**: `mpv`, `ffplay`, or `vlc` — the first one on `PATH`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[whisper,media]"
```

Installing both extras means every video is transcribed from audio — no silent
fallback to subtitles.

The first transcription downloads the Whisper `small` model (about 500 MB)
from Hugging Face and caches it; later runs are offline. It runs on CPU and
uses a CUDA GPU automatically when one is available.

## Quick start

```bash
# Web GUI (opens http://127.0.0.1:8765/)
vidsum gui

# Download + transcribe + summarize
vidsum run "https://www.youtube.com/watch?v=..."

# One-shot question about a processed video
vidsum ask <video_id_or_dir> "What did they say about X?"

# Threaded chat REPL (omit the question)
vidsum ask <video_id_or_dir>

# Terminal UI
vidsum tui
```

## CLI reference

| Command | What it does |
|---------|--------------|
| `vidsum run <url>` | Download, transcribe, summarize |
| `vidsum ask <ref> "<question>"` | One-shot grounded answer with citations |
| `vidsum ask <ref>` | Threaded chat REPL (`--new`, `--mode chat`, `--global`) |
| `vidsum threads <ref>` | List conversations for a video or the library |
| `vidsum list` | List processed videos |
| `vidsum models [--refresh] [--set <model>]` | Inspect or switch Ollama models |
| `vidsum delete <ref>` | Delete a processed video |
| `vidsum tui` / `vidsum gui` | Launch the terminal / web interface |
| `vidsum agents` | Run the agent layer |

`<ref>` is a video id, its data directory, or a URL. `run`, `ask`, `threads`,
`list`, and `models` accept `--json` for machine-readable output.

## Web GUI

- Paste a link on the left: choose **video / audio / subtitles**, force a redo,
  or use **subtitles only** to skip Whisper.
- A live progress bar tracks the background job, and a browser notification
  fires when it finishes (the server also sends a desktop notification via
  `notify-send`, so you can close the tab).
- The player streams from disk with seeking. Every timestamp in summaries,
  answers, and the transcript is clickable and jumps the player to that moment.
- The **Ask** tab is a persistent conversation: pick a thread, start a **New**
  one, delete one, and toggle **Ask** (strictly grounded, cited) vs **Chat**
  (conversational, history-aware). Follow-ups remember the whole conversation.
- The **Context** button shows how full the window is using Ollama's own
  numbers: the context loaded for the model, the model's maximum, and the last
  request's token counts.
- **Playlists** in the sidebar group related videos. **All videos · global
  chat** talks to the library as a whole, and its **Scope** picker restricts
  retrieval to selected playlists so unrelated topics never crowd the context.
- The top bar lists installed models, marks the one loaded in memory, and lets
  you switch or pull models without leaving the page.

## Transcription

1. Audio is transcribed with faster-whisper (default model `small`; override
   with `--whisper-model` or `VS_WHISPER_MODEL`).
2. `--subtitles` on the CLI or "subtitles only" in the GUI forces the subtitle
   path. `tiny`/`base` are faster, `small`+ is more accurate.

## Q&A and conversations

1. Chunks are embedded locally with `nomic-embed-text` via Ollama during
   processing (cached per video in `data/<video_id>/vectors.json`).
2. Top-k chunks are retrieved by cosine similarity.
3. Two modes share the retrieval:
   - **ask** — strictly grounded in the excerpts; every claim cites `[mm:ss]`
   - **chat** — conversational and history-aware; can connect ideas and use
     general knowledge, still citing the video when it draws on it.

Conversations are stored as threads in `data/chats/`. Every video has its own
threads, and **global** threads retrieve across the whole library (answers cite
`title @ [mm:ss]`), optionally scoped to playlists.

## Models

- Default chat model: `ornith-1.5:9b`
- Lighter options for low-end hardware: `qwen3:4b`, `qwen3:8b`, `gemma4:e2b`,
  or `gemma4:e2b-it-qat` (smallest quant)
- Embedding model: `nomic-embed-text`
- Switch anywhere: `vidsum models --set <model>`, the GUI dropdown, or the TUI
  `m` overlay
- Pull new models with `ollama pull <model>` or the GUI's **Pull…** button
- Env overrides: `VS_MODEL`, `VS_WHISPER_MODEL`, `VS_OLLAMA_URL`, `VS_DATA_DIR`

## Agent layer

```bash
vidsum agents                    # scripted safety demo (no LLM needed)
vidsum agents --agent analyst --task "Summarize the latest video"
vidsum agents --agent researcher --url "https://..." --interactive
```

Agents have **charters** (allowed tools and write paths), every tool has a
danger level, and destructive actions are denied unless `--interactive` or
`--auto` is passed. Every decision is appended to `data/agents/audit.jsonl`.

## Data layout

```
data/
  config.json               active model + settings
  chats/<thread_id>.json    conversation threads (video + global, with scope)
  playlists/<id>.json       playlists and their video ids
  jNQXAC9IVRw/
    metadata.json           title, author, duration, paths
    *.mp4 *.mp3 *.vtt       downloaded media
    transcript.json         text + timestamped segments
    chunks.json             retrieval/summary chunks
    summary.json / .md      summary + questions
    vectors.json            embedding cache
  agents/audit.jsonl        gate decisions
```

## Troubleshooting

- **"No video formats found" / 403s from YouTube** — update yt-dlp:
  `pip install -U yt-dlp` (the venv copy is preferred automatically).
- **"faster-whisper is not installed"** — reinstall with the extra:
  `pip install -e ".[whisper,media]"`.
- **"cannot reach Ollama"** — make sure `ollama serve` is running;
  `vidsum models --refresh` shows the live state.
- **"model not installed"** — `ollama pull ornith-1.5:9b` and
  `ollama pull nomic-embed-text`.
- **Model won't fit in memory** — switch to a lighter chat model:
  `ollama pull qwen3:4b` (or `gemma4:e2b-it-qat`), then
  `vidsum models --set qwen3:4b`.
- **Long videos** are summarized with map-reduce (per-section notes, then a
  final synthesis), so context limits are not a problem.
