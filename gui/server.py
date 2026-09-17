"""Local web GUI: stdlib HTTP server + REST API + media streaming with Range.

Routes
------
GET  /                          single-page app
GET  /static/<file>             app assets
GET  /api/health                server + model status
GET  /api/models                installed/loaded models + current selection
POST /api/models/current        switch the active model
POST /api/models/pull           pull a model (background job)
GET  /api/videos                list processed videos
POST /api/videos                process a URL (background job)
GET  /api/videos/<id>           full record (metadata, summary, chunks, threads)
DELETE /api/videos/<id>         delete a video and its artifacts
POST /api/videos/<id>/ask       ask a question (blocking, stateless)
GET  /api/videos/<id>/threads   list conversation threads for a video
POST /api/videos/<id>/threads   create a video conversation thread
GET  /api/threads               list global (library-wide) threads
POST /api/threads               create a global thread
GET  /api/threads/<id>          full thread with messages
PATCH /api/threads/<id>         edit title, mode, or playlist scope
DELETE /api/threads/<id>        delete a thread
POST /api/threads/<id>/messages append a message and get a reply (blocking)
GET  /api/playlists             list playlists with their video ids
POST /api/playlists             create a playlist
GET  /api/playlists/<id>        one playlist
PATCH /api/playlists/<id>       rename, add, or remove videos
DELETE /api/playlists/<id>      delete a playlist
GET  /api/context?thread_id=    context-window usage from Ollama
GET  /api/search?q=<query>      rank moments across every processed video
GET  /api/jobs/<id>             job status/progress
GET  /media/<id>               video or audio stream with HTTP Range
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from gui import notify
from pipeline import api, store
from pipeline.config import Config
from pipeline.llm import LLMError

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}
CHUNK_SIZE = 1024 * 256
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")
THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]{1,64}$")
PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]{1,64}$")


class JobManager:
    def __init__(self, notify: Callable[[str], None] | None = None) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._notify = notify

    def create(self, kind: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "message": "queued",
                "fraction": 0.0,
                "result": None,
                "error": "",
                "created": _now(),
            }
        return job_id

    def update(self, job_id: str, *, state: str | None = None,
               message: str | None = None, fraction: float | None = None,
               result: Any = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if state is not None:
                job["state"] = state
            if message is not None:
                job["message"] = message
            if fraction is not None:
                job["fraction"] = max(0.0, min(1.0, fraction))
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return json.loads(json.dumps(job)) if job else None

    def run(self, job_id: str, fn) -> None:
        def worker() -> None:
            self.update(job_id, state="running", message="starting", fraction=0.0)
            try:
                result = fn(lambda message, fraction: self.update(
                    job_id, message=message, fraction=fraction,
                ))
                self.update(
                    job_id, state="done", message="done", fraction=1.0, result=result,
                )
                self._notify_done(job_id)
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                self.update(
                    job_id, state="error", message=str(exc), error=str(exc),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _notify_done(self, job_id: str) -> None:
        if self._notify is None:
            return
        job = self.get(job_id) or {}
        result = job.get("result") or {}
        kind = job.get("kind")
        if kind == "process":
            label = result.get("title") or result.get("video_id") or "video"
            message = f"Finished processing: {label}"
        elif kind == "pull":
            message = f"Finished pulling {result.get('model', 'model')}"
        else:
            message = "Finished"
        try:
            self._notify(message)
        except Exception:  # noqa: BLE001 - notifications must never break jobs
            pass


class AppState:
    def __init__(self, out_dir: str, *, quiet: bool = False) -> None:
        self.out_dir = out_dir
        self.quiet = quiet
        self.jobs = JobManager(notify=None if quiet else self._desktop_notify)

    def _desktop_notify(self, message: str) -> None:
        cfg = self.config
        notify.desktop_notify("VideoSummarizer", message, enabled=cfg.desktop_notify)

    @property
    def config(self) -> Config:
        return Config.load(self.out_dir)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_join(base: Path, *parts: str) -> Path | None:
    candidate = base.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes=") or size <= 0:
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    try:
        if start_s == "":
            length = int(end_s)
            if length <= 0:
                return None
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    if start >= size or start < 0:
        return None
    return start, min(end, size - 1)


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VideoSummarizerGUI/0.2"
        protocol_version = "HTTP/1.1"

        # ---- helpers -------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:
            if state.quiet or self.path.startswith("/api/health"):
                return
            print(f"[gui] {self.address_string()} {fmt % args}")

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
            return payload if isinstance(payload, dict) else {}

        def _error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _serve_static(self, rel: str) -> None:
            path = _safe_join(STATIC_DIR, rel)
            if path is None or not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            body = path.read_bytes()
            content_type = MEDIA_TYPES.get(
                path.suffix.lower(),
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _serve_media(self, video_id: str) -> None:
            record = store.video_record(state.out_dir, video_id)
            if record is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown video")
                return
            media = None
            for key in ("video_path", "audio_path"):
                candidate = record.get(key)
                if candidate and Path(candidate).is_file():
                    media = Path(candidate)
                    break
            if media is None:
                self._error(HTTPStatus.NOT_FOUND, "no media downloaded")
                return
            self._stream_file(media)

        def _stream_file(self, path: Path) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                self._error(HTTPStatus.NOT_FOUND, "file vanished")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix.lower() in (".mkv", ".webm"):
                content_type = "video/webm"
            rng = parse_range(self.headers.get("Range"), size)
            if rng is None and self.headers.get("Range"):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        data = fh.read(min(CHUNK_SIZE, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ---- routing -------------------------------------------------

        def _route_get(self, path: str) -> None:
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(unquote(path[len("/static/"):]))
                return
            if path == "/api/health":
                status = api.models_status(config=state.config)
                self._send_json({"ok": True, "models": status, "time": _now()})
                return
            if path == "/api/models":
                status = api.models_status(config=state.config)
                self._send_json(status)
                return
            if path == "/api/videos":
                self._send_json(api.list_videos(out_dir=state.out_dir))
                return
            if path.startswith("/api/videos/"):
                rest = unquote(path[len("/api/videos/"):])
                if rest.endswith("/threads"):
                    video_id = rest[: -len("/threads")]
                    if not VIDEO_ID_RE.match(video_id):
                        self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                        return
                    self._send_json(api.list_threads(video_id, out_dir=state.out_dir))
                    return
                video_id = rest
                if not VIDEO_ID_RE.match(video_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                    return
                self._send_json(self._video_detail(video_id))
                return
            if path == "/api/threads":
                self._send_json(api.list_threads(out_dir=state.out_dir))
                return
            if path == "/api/playlists":
                self._send_json(api.list_playlists(out_dir=state.out_dir))
                return
            if path.startswith("/api/playlists/"):
                playlist_id = unquote(path[len("/api/playlists/"):])
                if not PLAYLIST_ID_RE.match(playlist_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad playlist id")
                    return
                self._send_json(api.get_playlist(playlist_id, out_dir=state.out_dir))
                return
            if path == "/api/context":
                params = parse_qs(urlparse(self.path).query)
                thread_id = params.get("thread_id", [""])[0].strip()
                self._send_json(
                    api.context_status(thread_id or None, out_dir=state.out_dir)
                )
                return
            if path == "/api/search":
                params = parse_qs(urlparse(self.path).query)
                query = params.get("q", [""])[0].strip()
                if not query:
                    self._error(HTTPStatus.BAD_REQUEST, "q is required")
                    return
                playlist_ids = [
                    p for p in params.get("playlists", [""])[0].split(",") if p
                ]
                self._send_json(
                    api.search_library(
                        query, out_dir=state.out_dir,
                        playlist_ids=playlist_ids or None,
                    )
                )
                return
            if path.startswith("/api/threads/"):
                thread_id = unquote(path[len("/api/threads/"):])
                if not THREAD_ID_RE.match(thread_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad thread id")
                    return
                self._send_json(api.get_thread(thread_id, out_dir=state.out_dir))
                return
            if path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                job = state.jobs.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown job")
                    return
                self._send_json(job)
                return
            if path.startswith("/media/"):
                video_id = unquote(path[len("/media/"):])
                if not VIDEO_ID_RE.match(video_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                    return
                self._serve_media(video_id)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _video_detail(self, video_id: str) -> dict[str, Any]:
            record = store.video_record(state.out_dir, video_id)
            if record is None:
                raise KeyError(video_id)
            transcript = store.load_transcript(state.out_dir, video_id)
            chunks = store.load_chunks(state.out_dir, video_id) or []
            record["transcript"] = transcript.to_dict() if transcript else None
            record["chunks"] = [c.to_dict() for c in chunks]
            store.migrate_answers(state.out_dir, video_id)
            record["answers"] = store.load_answers(state.out_dir, video_id)
            record["threads"] = store.list_threads(state.out_dir, video_id=video_id)
            record["media_url"] = (
                f"/media/{video_id}"
                if (record.get("has_video") or record.get("has_audio"))
                else None
            )
            return record

        def _route_post(self, path: str) -> None:
            payload = self._read_json()
            if path == "/api/models/current":
                model = str(payload.get("model") or "").strip()
                if not model:
                    self._error(HTTPStatus.BAD_REQUEST, "model is required")
                    return
                try:
                    self._send_json(api.set_model(model, data_dir=state.out_dir))
                except LLMError as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc))
                return
            if path == "/api/models/pull":
                model = str(payload.get("model") or "").strip()
                if not model:
                    self._error(HTTPStatus.BAD_REQUEST, "model is required")
                    return
                job_id = state.jobs.create("pull")
                cfg = state.config

                def pull(progress):
                    progress(f"pulling {model}", 0.05)
                    output = api.client_for(cfg).pull(model)
                    return {"model": model, "output": output[-2000:]}

                state.jobs.run(job_id, pull)
                self._send_json({"job_id": job_id}, status=202)
                return
            if path == "/api/videos":
                url = str(payload.get("url") or "").strip()
                if not url:
                    self._error(HTTPStatus.BAD_REQUEST, "url is required")
                    return
                job_id = state.jobs.create("process")
                options = {
                    "want_audio": bool(payload.get("audio", True)),
                    "want_video": bool(payload.get("video", True)),
                    "want_subtitles": bool(payload.get("subtitles", True)),
                    "force": bool(payload.get("force", False)),
                    "force_subtitles": bool(payload.get("force_subtitles", False)),
                    "model": payload.get("model") or None,
                    "whisper_model": payload.get("whisper_model") or None,
                }
                out_dir = state.out_dir

                def process(progress):
                    record = api.process(url, out_dir=out_dir, progress=progress, **options)
                    return {
                        "video_id": record.get("video_id"),
                        "title": record.get("title") or "",
                    }

                state.jobs.run(job_id, process)
                self._send_json({"job_id": job_id}, status=202)
                return
            if path.startswith("/api/videos/") and path.endswith("/threads"):
                video_id = unquote(path[len("/api/videos/"):-len("/threads")])
                if not VIDEO_ID_RE.match(video_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                    return
                try:
                    thread = api.create_thread(
                        video_id,
                        title=str(payload.get("title") or ""),
                        mode=str(payload.get("mode") or "ask"),
                        out_dir=state.out_dir,
                    )
                except LLMError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._send_json(thread, status=201)
                return
            if path == "/api/playlists":
                name = str(payload.get("name") or "").strip()
                if not name:
                    self._error(HTTPStatus.BAD_REQUEST, "name is required")
                    return
                self._send_json(
                    api.create_playlist(name, out_dir=state.out_dir), status=201,
                )
                return
            if path == "/api/threads":
                self._send_json(
                    api.create_thread(
                        None,
                        title=str(payload.get("title") or ""),
                        mode=str(payload.get("mode") or "ask"),
                        out_dir=state.out_dir,
                    ),
                    status=201,
                )
                return
            if path.startswith("/api/threads/") and path.endswith("/messages"):
                thread_id = unquote(path[len("/api/threads/"):-len("/messages")])
                if not THREAD_ID_RE.match(thread_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad thread id")
                    return
                message = str(payload.get("message") or "").strip()
                if not message:
                    self._error(HTTPStatus.BAD_REQUEST, "message is required")
                    return
                try:
                    result = api.post_message(
                        thread_id,
                        message,
                        mode=payload.get("mode") or None,
                        out_dir=state.out_dir,
                        top_k=int(payload["top_k"]) if payload.get("top_k") else None,
                        model=payload.get("model") or None,
                    )
                except LLMError as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc))
                    return
                self._send_json(result)
                return
            if path.startswith("/api/videos/") and path.endswith("/ask"):
                video_id = unquote(path[len("/api/videos/"):-len("/ask")])
                if not VIDEO_ID_RE.match(video_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                    return
                question = str(payload.get("question") or "").strip()
                if not question:
                    self._error(HTTPStatus.BAD_REQUEST, "question is required")
                    return
                try:
                    result = api.ask(
                        video_id,
                        question,
                        out_dir=state.out_dir,
                        top_k=int(payload.get("top_k") or state.config.top_k),
                        model=payload.get("model") or None,
                    )
                except LLMError as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc))
                    return
                self._send_json(result)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _route_patch(self, path: str) -> None:
            payload = self._read_json()
            if path.startswith("/api/threads/"):
                thread_id = unquote(path[len("/api/threads/"):])
                if not THREAD_ID_RE.match(thread_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad thread id")
                    return
                playlist_ids = payload.get("playlist_ids")
                if not isinstance(playlist_ids, list):
                    playlist_ids = None
                try:
                    thread = api.update_thread(
                        thread_id,
                        title=payload.get("title"),
                        playlist_ids=playlist_ids,
                        mode=payload.get("mode"),
                        out_dir=state.out_dir,
                    )
                except LLMError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._send_json(thread)
                return
            if path.startswith("/api/playlists/"):
                playlist_id = unquote(path[len("/api/playlists/"):])
                if not PLAYLIST_ID_RE.match(playlist_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad playlist id")
                    return
                add_ids = payload.get("add_video_ids")
                remove_ids = payload.get("remove_video_ids")
                try:
                    playlist = api.update_playlist(
                        playlist_id,
                        name=payload.get("name"),
                        add_video_ids=add_ids if isinstance(add_ids, list) else None,
                        remove_video_ids=(
                            remove_ids if isinstance(remove_ids, list) else None
                        ),
                        out_dir=state.out_dir,
                    )
                except LLMError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._send_json(playlist)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _route_delete(self, path: str) -> None:
            if path.startswith("/api/playlists/"):
                playlist_id = unquote(path[len("/api/playlists/"):])
                if not PLAYLIST_ID_RE.match(playlist_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad playlist id")
                    return
                deleted = api.delete_playlist(playlist_id, out_dir=state.out_dir)
                self._send_json({"deleted": deleted})
                return
            if path.startswith("/api/threads/"):
                thread_id = unquote(path[len("/api/threads/"):])
                if not THREAD_ID_RE.match(thread_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad thread id")
                    return
                deleted = api.delete_thread(thread_id, out_dir=state.out_dir)
                self._send_json({"deleted": deleted})
                return
            if path.startswith("/api/videos/"):
                video_id = unquote(path[len("/api/videos/"):])
                if not VIDEO_ID_RE.match(video_id):
                    self._error(HTTPStatus.BAD_REQUEST, "bad video id")
                    return
                deleted = api.delete_video(video_id, out_dir=state.out_dir)
                self._send_json({"deleted": deleted})
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        # ---- http verbs ----------------------------------------------

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._route_get(parsed.path)
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except LLMError as exc:
                self._error(HTTPStatus.BAD_GATEWAY, str(exc))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._route_post(parsed.path)
            except LLMError as exc:
                self._error(HTTPStatus.BAD_GATEWAY, str(exc))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_PATCH(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._route_patch(parsed.path)
            except LLMError as exc:
                self._error(HTTPStatus.BAD_GATEWAY, str(exc))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._route_delete(parsed.path)
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    return Handler


def run_server(
    *,
    out_dir: str = "data",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    state = AppState(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{port}/"
    print(f"vidsum GUI running at {url}  (data: {Path(out_dir).resolve()})")
    print("press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VideoSummarizer web GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_server(
        out_dir=args.out_dir, host=args.host, port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
