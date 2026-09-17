from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.model == "ornith-1.5:9b"
    assert cfg.whisper_model == "small"
    assert cfg.embed_model == "nomic-embed-text"


def test_config_save_and_load(tmp_data: Path):
    cfg = Config(data_dir=str(tmp_data), model="my-model", whisper_model="tiny")
    cfg.save()
    loaded = Config.load(str(tmp_data))
    assert loaded.model == "my-model"
    assert loaded.whisper_model == "tiny"
    assert loaded.data_dir == str(tmp_data)


def test_config_load_ignores_unknown_and_bad_json(tmp_data: Path):
    (tmp_data / "config.json").write_text("{broken")
    cfg = Config.load(str(tmp_data))
    assert cfg.model == "ornith-1.5:9b"

    (tmp_data / "config.json").write_text(json.dumps({"bogus": 1, "model": "m2"}))
    cfg2 = Config.load(str(tmp_data))
    assert cfg2.model == "m2"
    assert not hasattr(cfg2, "bogus")


def test_config_env_overrides(tmp_data: Path, monkeypatch):
    monkeypatch.setenv("VS_MODEL", "env-model")
    monkeypatch.setenv("VS_WHISPER_MODEL", "env-whisper")
    cfg = Config.load(str(tmp_data))
    assert cfg.model == "env-model"
    assert cfg.whisper_model == "env-whisper"
