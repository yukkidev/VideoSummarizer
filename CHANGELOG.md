# Changelog

## 2026-09-17 — Playlists, scoped global chat, context-window display

### Added

- **playlists**: named groups of videos stored in `data/playlists/<id>.json`
  with full CRUD (`api.create_playlist`, `update_playlist`,
  `list_playlists`, `delete_playlist`). Deleting a video removes it from every
  playlist; deleting a playlist never touches the library.
- **playlist scope for global chats**: threads carry `playlist_ids`; retrieval
  (and on-demand vector indexing) is restricted to the union of those
  playlists' videos, and the scope label is included in the prompt.
  `PATCH /api/threads/<id>` sets title, mode, and scope;
  `api.search_library(..., playlist_ids=...)` and
  `GET /api/search?q=...&playlists=...` expose scoped search.
- **GUI playlists panel** above the library: create, collapse/expand each
  playlist, add the current video (＋), remove videos (×), pick any library
  video from the expanded view's "Add video…" select.
- **scope picker** in the global chat toolbar: a checkbox dropdown of
  playlists with an "All videos" default; the button summarizes the current
  scope (`2 playlists · 7 videos`).
- **context-window display**: `/api/context?thread_id=` reads the context
  Ollama actually loaded (`/api/ps` `context_length`, e.g. 100,000), the
  model maximum (`/api/show` `*.context_length`), real per-reply token usage
  (`prompt_eval_count` / `eval_count` via `chat_detailed`), and a
  4-chars-per-token conversation estimate. The GUI shows a live `Context N%`
  button and a popup with a usage bar; replies persist their token counts.

### Changed

- `ChatMessage` stores `prompt_tokens`, `eval_tokens`, and `context_limit`;
  post-message responses include a `context` block.
- `OllamaClient.loaded_models()` now reports per-model `context_length`, and
  `generate_reply_detailed()` returns usage alongside the reply text.

## 2026-09-17 — Continuous chat, library-wide recall, notifications

### Added

- **chat threads**: persistent conversations in `data/chats/<id>.json` with
  titles, modes, and messages; per-video and global (library-wide) scopes with
  create/list/get/delete across the GUI, TUI, and CLI.
- **two reply modes**: `ask` (strictly transcript-grounded, timestamp-cited)
  and `chat` (conversational, history-aware). Each turn still retrieves
  sources, and the last ~12 turns are sent through Ollama's `/api/chat`.
- **library-wide retrieval**: `rag.search_all` / `api.search_library` rank
  chunks from every processed video and label citations with video title +
  timestamp. Vectors are now built during `process`, so new videos are
  immediately searchable; `GET /api/search?q=` exposes the search over HTTP.
- **notifications**: browser Notification API when a GUI job finishes, plus a
  server-side `notify-send` desktop notification (`desktop_notify` config,
  default on) that works even with the tab closed.
- **GUI**: thread picker with New/Delete, Ask/Chat toggle, cross-video citation
  chips that jump to the cited video, and an "All videos · global chat" entry.
- **CLI/TUI**: `vidsum threads`, `ask --thread/--new/--mode/--global`, threaded
  REPL commands (`/mode`, `/new`, `/threads`, `/thread`); TUI keys `n` (new
  thread), `t` (cycle), `M` (toggle mode). Agent tool `search_library`.
- **legacy migration**: `answers.jsonl` becomes an "Earlier questions" thread
  once, then is renamed to `answers.jsonl.migrated`.

### Changed

- `api.ask` is now stateless (one-shot); persistent Q&A goes through
  `api.post_message` and threads. Deleting a video deletes its threads.
- `Citation` carries optional `video_id` / `video_title` for cross-video hits.

### Fixed

- Q&A no longer loses context between turns: replies are generated from the
  thread history instead of a single fresh prompt.

## 2026-09-17 — Full build-out from skeleton

### Added

- **pipeline**: yt-dlp downloader (metadata via `-J`, deterministic audio/video
  paths, manual + automatic subtitles), faster-whisper transcription with
  subtitle fallback, segment-aware chunking, map-reduce summarization,
  per-video artifact store, and an `api` facade shared by every interface.
- **interfaces**: stdlib web GUI (Range-streamed video player, clickable
  timestamp citations, built-in downloader form, model switcher with live
  loaded-in-VRAM state, background jobs with progress) and a curses TUI
  (library browser, tabs, ask prompt, model overlay, external player at
  timestamp).
- **rag**: local embeddings (`nomic-embed-text`) + top-k retrieval Q&A with
  timestamp-cited answers; vectors cached per video.
- **agents**: real Ollama tool loop with enforced charters, danger-leveled
  tools, interactive/auto `HumanGate`, repeated-call detection, error feedback,
  and a JSONL audit log.
- **cli**: `vidsum run|ask|list|models|delete|tui|gui|agents` with JSON output.
- **testing**: 218 tests, including mocked HTTP/subprocess suites and live
  markers for Ollama, Whisper, and network paths; ruff config.

### Changed

- Default Ollama model is `ornith-1.5:9b`, selectable everywhere and persisted
  in `data/config.json`; `VS_MODEL` / `VS_WHISPER_MODEL` / `VS_OLLAMA_URL`
  environment overrides.
- `python -m pipeline.runner` retained as a thin shim over the new API.

### Fixed

- Skeleton-fatal bugs: undefined `subs_available`, broken `--json` kwarg,
  unusable `--no-audio` path, invalid whisper invocation, paragraph chunking
  order, undefined `researcher` in the demo, and an inert interactive
  `HumanGate`.
- yt-dlp resolution now prefers a venv-local binary (old system yt-dlp fails
  on current YouTube).
