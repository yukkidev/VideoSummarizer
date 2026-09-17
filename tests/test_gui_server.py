from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import gui.server as gui
from gui.server import AppState, make_handler, parse_range
from pipeline import store
from pipeline.models import Chunk, Segment, Summary, Transcript, Video


def request(url: str, method: str = "GET", payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


@pytest.fixture()
def seeded(tmp_data: Path):
    vdir = tmp_data / "vid1"
    vdir.mkdir(parents=True, exist_ok=True)
    video = Video(
        video_id="vid1",
        url="https://example.com/vid1",
        title="Test Video",
        author="Author",
        duration=120.0,
        video_path=str(vdir / "vid1.mp4"),
        audio_path=str(vdir / "vid1.mp3"),
        subtitle_path=str(vdir / "vid1.en.vtt"),
    )
    (vdir / "vid1.mp4").write_bytes(b"FAKEVIDEO" * 20)
    (vdir / "vid1.mp3").write_bytes(b"FAKEAUDIO")
    (vdir / "vid1.en.vtt").write_text("WEBVTT")
    store.save_video(tmp_data, video)
    store.save_transcript(
        tmp_data, "vid1",
        Transcript(
            text="Hello world. Second sentence.",
            segments=[Segment(0.0, 5.0, "Hello world."), Segment(5.0, 9.0, "Second sentence.")],
            source="faster-whisper/tiny",
            language="en",
            duration=9.0,
        ),
    )
    store.save_chunks(
        tmp_data, "vid1",
        [Chunk(0, 0.0, 5.0, "Hello world."), Chunk(1, 5.0, 9.0, "Second sentence.")],
    )
    store.save_summary(
        tmp_data, "vid1",
        Summary(video_id="vid1", title="Test Video", model="m",
                summary="- A point", highlights=["Why?"]),
    )
    store.append_answer(
        tmp_data,
        __import__("pipeline.models", fromlist=["Answer"]).Answer(
            video_id="vid1", question="q?", answer="a [00:05]", model="m",
        ),
        created="2026-01-01T00:00:00Z",
    )
    return tmp_data


@pytest.fixture()
def server(seeded: Path):
    state = AppState(str(seeded), quiet=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_parse_range_variants():
    assert parse_range("bytes=0-3", 10) == (0, 3)
    assert parse_range("bytes=5-", 10) == (5, 9)
    assert parse_range("bytes=-4", 10) == (6, 9)
    assert parse_range("bytes=0-999", 10) == (0, 9)
    assert parse_range("bytes=99-", 10) is None
    assert parse_range(None, 10) is None
    assert parse_range("items=0-1", 10) is None
    assert parse_range("bytes=abc", 10) is None


def test_safe_join_blocks_traversal(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    assert gui._safe_join(base, "ok.txt") is not None
    assert gui._safe_join(base, "..", "outside.txt") is None


def test_index_served(server):
    status, headers, body = request(f"{server}/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"VideoSummarizer" in body


def test_static_asset(server):
    status, headers, body = request(f"{server}/static/app.js")
    assert status == 200
    assert "javascript" in headers["Content-Type"]
    assert b"refreshModels" in body


def test_api_health(server, monkeypatch):
    monkeypatch.setattr(
        gui.api, "models_status",
        lambda model=None, config=None: {
            "server_ok": True, "current": "m", "current_loaded": True,
            "installed": [{"name": "m"}], "loaded": [], "error": "",
        },
    )
    status, _, body = request(f"{server}/api/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["models"]["current"] == "m"


def test_api_videos_list(server):
    status, _, body = request(f"{server}/api/videos")
    assert status == 200
    videos = json.loads(body)
    assert videos[0]["video_id"] == "vid1"
    assert videos[0]["has_video"] is True


def test_api_video_detail(server):
    status, _, body = request(f"{server}/api/videos/vid1")
    assert status == 200
    record = json.loads(body)
    assert record["title"] == "Test Video"
    assert len(record["chunks"]) == 2
    assert record["answers"] == []
    assert record["threads"][0]["title"] == "Earlier questions"
    assert record["threads"][0]["message_count"] == 2
    assert record["media_url"] == "/media/vid1"
    assert record["transcript"]["source"] == "faster-whisper/tiny"


def test_api_video_detail_missing(server):
    status, _, body = request(f"{server}/api/videos/nope")
    assert status == 404
    assert json.loads(body)["error"]


def test_api_video_detail_bad_id(server):
    status, _, _ = request(f"{server}/api/videos/..%2fetc")
    assert status in (400, 404)


def test_media_full_and_range(server):
    status, headers, body = request(f"{server}/media/vid1")
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    full_size = len(body)
    assert full_size > 100

    status, headers, body = request(
        f"{server}/media/vid1", headers={"Range": "bytes=0-3"},
    )
    assert status == 206
    assert headers["Content-Range"] == f"bytes 0-3/{full_size}"
    assert body == b"FAKE"

    status, headers, body = request(
        f"{server}/media/vid1", headers={"Range": f"bytes={full_size - 4}-"},
    )
    assert status == 206
    assert body == b"IDEO"


def test_media_bad_range(server):
    status, headers, _ = request(
        f"{server}/media/vid1", headers={"Range": "bytes=999999-"},
    )
    assert status == 416
    assert headers["Content-Range"].startswith("bytes */")


def test_media_missing(server):
    status, _, _ = request(f"{server}/media/nope")
    assert status == 404


def test_delete_video(server, seeded: Path):
    status, _, body = request(f"{server}/api/videos/vid1", method="DELETE")
    assert status == 200
    assert json.loads(body)["deleted"] is True
    assert not (seeded / "vid1").exists()


def test_process_job(server, monkeypatch):
    def fake_process(url, out_dir=None, progress=None, **kwargs):
        if progress:
            progress("downloading", 0.5)
        return {"video_id": "vid9", "title": "New"}

    monkeypatch.setattr(gui.api, "process", fake_process)
    status, _, body = request(
        f"{server}/api/videos", method="POST", payload={"url": "https://example.com/x"},
    )
    assert status == 202
    job_id = json.loads(body)["job_id"]

    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        _, _, job_body = request(f"{server}/api/jobs/{job_id}")
        job = json.loads(job_body)
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "done"
    assert job["result"]["video_id"] == "vid9"
    assert job["fraction"] == 1.0


def test_process_job_requires_url(server):
    status, _, _ = request(f"{server}/api/videos", method="POST", payload={})
    assert status == 400


def test_process_job_error(server, monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("download exploded")

    monkeypatch.setattr(gui.api, "process", boom)
    _, _, body = request(
        f"{server}/api/videos", method="POST", payload={"url": "https://example.com/x"},
    )
    job_id = json.loads(body)["job_id"]
    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        _, _, job_body = request(f"{server}/api/jobs/{job_id}")
        job = json.loads(job_body)
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    assert "exploded" in job["error"]


def test_ask_endpoint(server, monkeypatch):
    def fake_ask(ref, question, **kwargs):
        return {"answer": {"question": question, "answer": "because [00:05]", "citations": []}}

    monkeypatch.setattr(gui.api, "ask", fake_ask)
    status, _, body = request(
        f"{server}/api/videos/vid1/ask", method="POST", payload={"question": "why?"},
    )
    assert status == 200
    assert json.loads(body)["answer"]["answer"].startswith("because")


def test_ask_endpoint_requires_question(server):
    status, _, _ = request(
        f"{server}/api/videos/vid1/ask", method="POST", payload={},
    )
    assert status == 400


def test_switch_model_endpoint(server, monkeypatch):
    seen = {}

    def fake_set(model, data_dir=None):
        seen["model"] = model
        return {"server_ok": True, "current": model, "current_loaded": False,
                "installed": [], "loaded": [], "error": ""}

    monkeypatch.setattr(gui.api, "set_model", fake_set)
    status, _, body = request(
        f"{server}/api/models/current", method="POST", payload={"model": "new-m"},
    )
    assert status == 200
    assert seen["model"] == "new-m"
    assert json.loads(body)["current"] == "new-m"


def test_pull_model_job(server, monkeypatch):
    class FakeClient:
        def pull(self, model):
            return "pulled " + model

    monkeypatch.setattr(gui.api, "client_for", lambda cfg=None: FakeClient())
    status, _, body = request(
        f"{server}/api/models/pull", method="POST", payload={"model": "tiny"},
    )
    assert status == 202
    job_id = json.loads(body)["job_id"]
    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        _, _, job_body = request(f"{server}/api/jobs/{job_id}")
        job = json.loads(job_body)
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "done"
    assert job["result"]["model"] == "tiny"


def test_unknown_job(server):
    status, _, _ = request(f"{server}/api/jobs/doesnotexist")
    assert status == 404


def test_unknown_route(server):
    status, _, _ = request(f"{server}/nope")
    assert status == 404


def test_create_and_list_video_threads(server):
    status, _, body = request(
        f"{server}/api/videos/vid1/threads", method="POST", payload={"mode": "chat"},
    )
    assert status == 201
    thread = json.loads(body)
    assert thread["scope"] == "video"
    assert thread["video_id"] == "vid1"
    assert thread["mode"] == "chat"

    status, _, body = request(f"{server}/api/videos/vid1/threads")
    assert status == 200
    threads = json.loads(body)
    titles = [t["title"] for t in threads]
    assert "Earlier questions" in titles
    assert any(t["thread_id"] == thread["thread_id"] for t in threads)


def test_global_thread_create_message_and_delete(server, monkeypatch):
    status, _, body = request(f"{server}/api/threads", method="POST", payload={"mode": "chat"})
    assert status == 201
    thread = json.loads(body)
    assert thread["scope"] == "global"
    assert thread["video_id"] == ""

    def fake_post(thread_id, message, **kwargs):
        return {
            "thread": {"thread_id": thread_id, "title": message, "messages": []},
            "message": {"role": "assistant", "content": "hi", "citations": []},
        }

    monkeypatch.setattr(gui.api, "post_message", fake_post)
    status, _, body = request(
        f"{server}/api/threads/{thread['thread_id']}/messages",
        method="POST", payload={"message": "hello"},
    )
    assert status == 200
    assert json.loads(body)["message"]["content"] == "hi"

    status, _, body = request(f"{server}/api/threads/{thread['thread_id']}")
    assert status == 200
    assert json.loads(body)["thread_id"] == thread["thread_id"]

    status, _, body = request(
        f"{server}/api/threads/{thread['thread_id']}", method="DELETE",
    )
    assert json.loads(body)["deleted"] is True
    status, _, _ = request(f"{server}/api/threads/{thread['thread_id']}")
    assert status == 502


def test_thread_message_requires_message(server):
    created = json.loads(
        request(f"{server}/api/threads", method="POST", payload={})[2]
    )
    status, _, _ = request(
        f"{server}/api/threads/{created['thread_id']}/messages",
        method="POST", payload={},
    )
    assert status == 400


def test_search_endpoint(server, monkeypatch):
    seen = {}

    def fake_search(query, out_dir=None, top_k=None, playlist_ids=None):
        seen["playlist_ids"] = playlist_ids
        return {
            "query": query, "citations": [{"video_id": "vid1", "start": 0.0}],
        }

    monkeypatch.setattr(gui.api, "search_library", fake_search)
    status, _, body = request(f"{server}/api/search?q=hello&playlists=p1,p2")
    assert status == 200
    payload = json.loads(body)
    assert payload["query"] == "hello"
    assert payload["citations"][0]["video_id"] == "vid1"
    assert seen["playlist_ids"] == ["p1", "p2"]

    status, _, _ = request(f"{server}/api/search")
    assert status == 400


def test_playlist_endpoints(server):
    status, _, body = request(
        f"{server}/api/playlists", method="POST", payload={"name": "Physics"},
    )
    assert status == 201
    playlist = json.loads(body)
    playlist_id = playlist["playlist_id"]
    assert playlist["video_count"] == 0

    status, _, body = request(
        f"{server}/api/playlists/{playlist_id}", method="PATCH",
        payload={"add_video_ids": ["vid1"]},
    )
    assert status == 200
    updated = json.loads(body)
    assert updated["video_ids"] == ["vid1"]
    assert updated["video_count"] == 1

    status, _, body = request(f"{server}/api/playlists")
    assert json.loads(body)[0]["name"] == "Physics"

    status, _, body = request(
        f"{server}/api/playlists/{playlist_id}", method="PATCH",
        payload={"remove_video_ids": ["vid1"], "name": "Quantum"},
    )
    assert json.loads(body)["video_ids"] == []
    assert json.loads(body)["name"] == "Quantum"

    status, _, body = request(f"{server}/api/playlists/{playlist_id}", method="DELETE")
    assert json.loads(body)["deleted"] is True

    status, _, _ = request(f"{server}/api/playlists", method="POST", payload={})
    assert status == 400


def test_thread_patch_scope_and_mode(server):
    created = json.loads(
        request(f"{server}/api/threads", method="POST", payload={})[2]
    )
    playlist = json.loads(
        request(f"{server}/api/playlists", method="POST", payload={"name": "P"})[2]
    )
    status, _, body = request(
        f"{server}/api/threads/{created['thread_id']}", method="PATCH",
        payload={"playlist_ids": [playlist["playlist_id"]], "mode": "chat", "title": "Scoped"},
    )
    assert status == 200
    thread = json.loads(body)
    assert thread["playlist_ids"] == [playlist["playlist_id"]]
    assert thread["mode"] == "chat"
    assert thread["title"] == "Scoped"


def test_context_endpoint(server, monkeypatch):
    monkeypatch.setattr(
        gui.api, "context_status",
        lambda thread_id=None, out_dir=None: {
            "model": "m",
            "loaded_context": 100000,
            "max_context": 262144,
            "thread_id": thread_id or "",
        },
    )
    status, _, body = request(f"{server}/api/context?thread_id=t1")
    assert status == 200
    payload = json.loads(body)
    assert payload["loaded_context"] == 100000
    assert payload["thread_id"] == "t1"


def test_job_manager_notifies_on_done(seeded: Path):
    sent: list[str] = []
    jobs = gui.JobManager(notify=sent.append)
    job_id = jobs.create("process")
    jobs.run(job_id, lambda progress: {"video_id": "v9", "title": "New Video"})
    deadline = time.time() + 5
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["state"] == "done" and sent:
            break
        time.sleep(0.02)
    assert jobs.get(job_id)["state"] == "done"
    assert sent == ["Finished processing: New Video"]


def test_quiet_app_state_disables_notifications(seeded: Path):
    state = AppState(str(seeded), quiet=True)
    assert state.jobs._notify is None
