"""Persistent application settings, stored in data/config.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_MODEL = "ornith-1.5:9b"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_DATA_DIR = "data"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    data_dir: str = DEFAULT_DATA_DIR
    device: str = "auto"
    language: str = ""
    top_k: int = 6

    @property
    def path(self) -> Path:
        return Path(self.data_dir) / "config.json"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, data_dir: str | None = None) -> Config:
        env_model = os.environ.get("VS_MODEL")
        env_data = os.environ.get("VS_DATA_DIR") or data_dir or DEFAULT_DATA_DIR
        cfg = cls(data_dir=env_data)
        p = Path(env_data) / "config.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                raw = {}
            known = {f.name for f in fields(cls)}
            for key, value in raw.items():
                if key in known:
                    setattr(cfg, key, value)
        cfg.data_dir = env_data
        if env_model:
            cfg.model = env_model
        env_whisper = os.environ.get("VS_WHISPER_MODEL")
        if env_whisper:
            cfg.whisper_model = env_whisper
        env_ollama = os.environ.get("VS_OLLAMA_URL")
        if env_ollama:
            cfg.ollama_url = env_ollama
        return cfg
