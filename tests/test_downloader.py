from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import pipeline.downloader as dl
from pipeline.downloader import (
    DownloaderError,
    DownloadPlan,
    available_subtitle_langs,
    choose_subtitle_lang,
    fetch,
    fetch_info,
    find_subtitle_file,
    run_yt_dlp,
    video_from_info,
)
from tests.conftest import completed

INFO = {
    "id": "vid123",
    "title": "My Video",
    "uploader": "Great Channel",
    "duration": 123.0,
    "description": "A description",
    "webpage_url": "https://example.com/watch?v=vid123",
    "thumbnail": "https://example.com/t.jpg",
    "subtitles": {"en": [{"ext": "vtt"}], "fr": [{"ext": "vtt"}]},
    "automatic_captions": {"de": [{"ext": "vtt"}], "en-orig": [{"ext": "vtt"}]},
}


def test_run_yt_dlp_raises_on_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed("", "boom", 1))
    with pytest.raises(DownloaderError, match="boom"):
        run_yt_dlp(["-J", "url"])


def test_run_yt_dlp_includes_required_flags(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return completed("{}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_yt_dlp(["-J", "u"])
    assert seen["cmd"][0].endswith("yt-dlp")
    assert "--no-warnings" in seen["cmd"]
    assert "--no-playlist" in seen["cmd"]


def test_run_yt_dlp_missing_binary(monkeypatch):
    def fake_run(*a, **k):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(DownloaderError, match="not installed"):
        run_yt_dlp([])


def test_run_yt_dlp_timeout(monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(DownloaderError, match="timed out"):
        run_yt_dlp([], timeout=5)


def test_fetch_info_parses_json(monkeypatch):
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed(json.dumps(INFO)))
    info = fetch_info("https://example.com/watch?v=vid123")
    assert info["id"] == "vid123"


def test_fetch_info_picks_first_playlist_entry(monkeypatch):
    payload = {"_type": "playlist", "entries": [INFO]}
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed(json.dumps(payload)))
    assert fetch_info("u")["id"] == "vid123"


def test_fetch_info_empty_playlist_errors(monkeypatch):
    payload = {"_type": "playlist", "entries": []}
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed(json.dumps(payload)))
    with pytest.raises(DownloaderError, match="playlist"):
        fetch_info("u")


def test_fetch_info_rejects_garbage(monkeypatch):
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed("not json"))
    with pytest.raises(DownloaderError, match="parse"):
        fetch_info("u")


def test_video_from_info_duration_none():
    v = video_from_info({"id": "a", "title": "x", "duration": None})
    assert v.duration is None


def test_available_subtitle_langs_prioritizes_english():
    langs = available_subtitle_langs(INFO)
    assert langs[0].startswith("en")
    assert set(langs) == {"en", "fr", "de", "en-orig"}


def test_choose_subtitle_lang_prefers_manual_preferred():
    assert choose_subtitle_lang(INFO, "fr") == "fr"


def test_choose_subtitle_lang_prefers_manual_english_over_auto():
    assert choose_subtitle_lang(INFO, "") == "en"


def test_choose_subtitle_lang_falls_back_to_auto():
    info = {"subtitles": {}, "automatic_captions": {"de": [], "en": []}}
    assert choose_subtitle_lang(info) == "en"


def test_choose_subtitle_lang_none_when_empty():
    assert choose_subtitle_lang({}) is None


def test_find_subtitle_file(tmp_path: Path):
    (tmp_path / "vid123.en.vtt").write_text("WEBVTT")
    assert find_subtitle_file(tmp_path, "vid123").endswith("vid123.en.vtt")
    assert find_subtitle_file(tmp_path, "other") is None


def test_fetch_end_to_end_with_mocked_ytdlp(monkeypatch, tmp_data: Path):
    calls: list[list[str]] = []

    def fake_ytdlp(args, **kwargs):
        calls.append(args)
        if "-J" in args:
            return completed(json.dumps(INFO))
        vdir = tmp_data / "vid123"
        vdir.mkdir(parents=True, exist_ok=True)
        if "--skip-download" in args:
            path = vdir / "vid123.en.vtt"
            path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n")
            return completed(str(path) + "\n")
        if "-x" in args:
            path = vdir / "vid123.mp3"
            path.write_bytes(b"ID3fake")
            return completed(f"some log\n{path}\n")
        path = vdir / "vid123.mp4"
        path.write_bytes(b"fakevideo")
        return completed(f"{path}\n")

    monkeypatch.setattr(dl, "run_yt_dlp", fake_ytdlp)
    video = fetch(INFO["webpage_url"], tmp_data, plan=DownloadPlan())

    assert video.video_id == "vid123"
    assert video.subtitle_path and video.subtitle_path.endswith(".vtt")
    assert video.audio_path and video.audio_path.endswith(".mp3")
    assert video.video_path and video.video_path.endswith(".mp4")
    assert (tmp_data / "vid123" / "metadata.json").exists()
    meta = json.loads((tmp_data / "vid123" / "metadata.json").read_text())
    assert meta["title"] == "My Video"


def test_fetch_no_subtitles_available(monkeypatch, tmp_data: Path):
    info = {**INFO, "subtitles": {}, "automatic_captions": {}}
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed(json.dumps(info)))
    video = fetch(INFO["webpage_url"], tmp_data, plan=DownloadPlan(
        want_audio=False, want_video=False, want_subtitles=True,
    ))
    assert video.subtitle_path is None


def test_download_audio_falls_back_to_glob(monkeypatch, tmp_data: Path):
    def fake_ytdlp(args, **kwargs):
        vdir = tmp_data / "vid123"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "vid123.mp3").write_bytes(b"x")
        return completed("log without path\n")

    monkeypatch.setattr(dl, "run_yt_dlp", fake_ytdlp)
    path = dl.download_audio("u", tmp_data / "vid123", video_id="vid123")
    assert path.endswith("vid123.mp3")


def test_download_video_missing_output_raises(monkeypatch, tmp_data: Path):
    monkeypatch.setattr(dl, "run_yt_dlp", lambda *a, **k: completed("no path\n"))
    with pytest.raises(DownloaderError, match="no output file"):
        dl.download_video("u", tmp_data / "vid123", video_id="vid123")
