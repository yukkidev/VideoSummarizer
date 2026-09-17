from __future__ import annotations

import json

import pytest

import cli.main as cli
from pipeline.llm import LLMError
from pipeline.models import Answer, Citation


def test_parser_run_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "https://example.com/v"])
    assert args.command == "run"
    assert args.out_dir == "data"
    assert args.no_audio is False


def test_parser_out_dir_after_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "https://example.com/v", "--out-dir", "/tmp/x"])
    assert args.out_dir == "/tmp/x"


def test_parser_rejects_missing_command():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cmd_list_empty(tmp_path, capsys):
    args = cli.build_parser().parse_args(["list", "--out-dir", str(tmp_path)])
    assert cli.cmd_list(args) == 0
    assert "no videos" in capsys.readouterr().out


def test_cmd_list_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli.api, "list_videos", lambda out_dir=None: [{"video_id": "v1", "title": "T"}]
    )
    args = cli.build_parser().parse_args(["list", "--json", "--out-dir", str(tmp_path)])
    assert cli.cmd_list(args) == 0
    assert json.loads(capsys.readouterr().out)[0]["video_id"] == "v1"


def test_cmd_run_json(monkeypatch, capsys):
    record = {"video_id": "v1", "title": "T", "summary": {"summary": "s", "highlights": []}}
    monkeypatch.setattr(cli.api, "process", lambda url, **kwargs: record)
    args = cli.build_parser().parse_args(["run", "https://example.com/v", "--json"])
    assert cli.cmd_run(args) == 0
    assert json.loads(capsys.readouterr().out)["video_id"] == "v1"


def test_cmd_run_error(monkeypatch, capsys):
    def boom(url, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli.api, "process", boom)
    args = cli.build_parser().parse_args(["run", "https://example.com/v"])
    assert cli.cmd_run(args) == 1
    assert "download failed" in capsys.readouterr().err


def test_cmd_ask_one_shot(monkeypatch, capsys):
    payload = {
        "answer": Answer(
            video_id="v1", question="q", answer="a",
            citations=[Citation(0, 5.0, 6.0, "text", 0.9)],
        ).to_dict(),
        "video": {"video_path": "/tmp/v.mp4"},
    }
    monkeypatch.setattr(cli.api, "ask", lambda *a, **k: payload)
    args = cli.build_parser().parse_args(["ask", "v1", "what", "happened"])
    assert cli.cmd_ask(args) == 0
    out = capsys.readouterr().out
    assert "A: a" in out
    assert "[00:05]" in out


def test_cmd_ask_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise LLMError("no video")

    monkeypatch.setattr(cli.api, "ask", boom)
    args = cli.build_parser().parse_args(["ask", "v1", "q"])
    assert cli.cmd_ask(args) == 1
    assert "no video" in capsys.readouterr().err


def test_cmd_models_json(monkeypatch, capsys):
    status = {
        "server_ok": True,
        "current": "m",
        "current_loaded": True,
        "installed": [{"name": "m", "size_gb": 1.0}],
        "loaded": [],
        "error": "",
    }
    monkeypatch.setattr(cli.api, "load_config", lambda out: None)
    monkeypatch.setattr(cli.api, "models_status", lambda *a, **k: status)
    args = cli.build_parser().parse_args(["models", "--json"])
    assert cli.cmd_models(args) == 0
    assert json.loads(capsys.readouterr().out)["current"] == "m"


def test_cmd_models_set(monkeypatch, capsys):
    seen = {}

    def fake_set(model, data_dir=None):
        seen["model"] = model
        return {"server_ok": True, "current": model, "installed": [], "loaded": []}

    monkeypatch.setattr(cli.api, "set_model", fake_set)
    args = cli.build_parser().parse_args(["models", "--set", "new-model"])
    assert cli.cmd_models(args) == 0
    assert seen["model"] == "new-model"


def test_main_dispatches(monkeypatch):
    called = {}

    def fake_process(url, **kwargs):
        called["url"] = url
        return {"video_id": "v1", "title": "T", "summary": {}}

    monkeypatch.setattr(cli.api, "process", fake_process)
    assert cli.main(["run", "https://example.com/v", "--quiet"]) == 0
    assert called["url"] == "https://example.com/v"
