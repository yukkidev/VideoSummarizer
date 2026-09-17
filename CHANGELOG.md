# Changelog

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
