"""Ollama client: model listing/refresh, generation, embeddings, pulls.

Uses only the standard library (urllib) against the local Ollama HTTP API.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pipeline.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    Config,
)

THINK_PAIRED_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think\s*>")
THINK_OPEN_RE = re.compile(r"(?is)<think\b[^>]*>")
THINK_CLOSE_RE = re.compile(r"(?is)</think\s*>")

DEFAULT_GENERATE_TIMEOUT = 900
DEFAULT_EMBED_TIMEOUT = 180
DEFAULT_LIST_TIMEOUT = 15


class LLMError(RuntimeError):
    pass


def strip_thinking(text: str) -> str:
    text = text or ""
    text = THINK_PAIRED_RE.sub("", text)
    if THINK_OPEN_RE.search(text):
        text = THINK_OPEN_RE.split(text)[0]
    if THINK_CLOSE_RE.search(text):
        text = THINK_CLOSE_RE.split(text)[-1]
    return text.replace("\r\n", "\n").strip()


@dataclass
class InstalledModel:
    name: str
    size: int = 0
    modified_at: str = ""
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "size_gb": round(self.size / 1e9, 2),
            "modified_at": self.modified_at,
            "digest": self.digest,
        }


@dataclass
class LoadedModel:
    name: str
    size_vram: int = 0
    expires_at: str = ""
    context_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_vram": self.size_vram,
            "size_vram_gb": round(self.size_vram / 1e9, 2),
            "expires_at": self.expires_at,
            "context_length": self.context_length,
        }


@dataclass
class ModelStatus:
    server_ok: bool = False
    current: str = ""
    installed: list[InstalledModel] = field(default_factory=list)
    loaded: list[LoadedModel] = field(default_factory=list)
    error: str = ""

    @property
    def current_loaded(self) -> bool:
        return any(m.name == self.current for m in self.loaded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_ok": self.server_ok,
            "current": self.current,
            "current_loaded": self.current_loaded,
            "installed": [m.to_dict() for m in self.installed],
            "loaded": [m.to_dict() for m in self.loaded],
            "error": self.error,
        }


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: int = DEFAULT_GENERATE_TIMEOUT,
        generate: Callable[..., str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self._generate_override = generate

    # ---- transport -----------------------------------------------------

    def _request(self, path: str, payload: dict | None, *, timeout: int) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise LLMError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.base_url} ({exc.reason}). "
                "Is `ollama serve` running?"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(f"Ollama request to {path} timed out after {timeout}s") from exc
        try:
            return json.loads(body or "{}")
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON from Ollama {path}") from exc

    # ---- models --------------------------------------------------------

    def health(self) -> bool:
        try:
            self._request("/api/version", None, timeout=DEFAULT_LIST_TIMEOUT)
            return True
        except LLMError:
            return False

    def list_models(self) -> list[InstalledModel]:
        data = self._request("/api/tags", None, timeout=DEFAULT_LIST_TIMEOUT)
        models = []
        for raw in data.get("models", []):
            models.append(
                InstalledModel(
                    name=raw.get("name") or raw.get("model") or "",
                    size=int(raw.get("size") or 0),
                    modified_at=raw.get("modified_at") or "",
                    digest=raw.get("digest") or "",
                )
            )
        return models

    def loaded_models(self) -> list[LoadedModel]:
        data = self._request("/api/ps", None, timeout=DEFAULT_LIST_TIMEOUT)
        loaded = []
        for raw in data.get("models", []):
            loaded.append(
                LoadedModel(
                    name=raw.get("name") or raw.get("model") or "",
                    size_vram=int(raw.get("size_vram") or 0),
                    expires_at=raw.get("expires_at") or "",
                    context_length=int(raw.get("context_length") or 0),
                )
            )
        return loaded

    def show_model(self, model: str | None = None) -> dict[str, Any]:
        return self._request(
            "/api/show", {"model": model or self.model}, timeout=DEFAULT_LIST_TIMEOUT,
        )

    def model_context_length(self, model: str | None = None) -> int:
        """Maximum context the model was trained with, from /api/show."""
        name = model or self.model
        try:
            data = self.show_model(name)
        except LLMError:
            return 0
        info = data.get("model_info") or {}
        arch = str(info.get("general.architecture") or "")
        for key in (f"{arch}.context_length", "general.context_length", "context_length"):
            value = info.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        for key, value in info.items():
            if key.endswith(".context_length") and isinstance(value, (int, float)) and value > 0:
                return int(value)
        match = re.search(r"num_ctx\s+(\d+)", str(data.get("parameters") or ""))
        return int(match.group(1)) if match else 0

    def loaded_context_length(self, model: str | None = None) -> int:
        """Context Ollama actually allocated for the loaded model, from /api/ps."""
        name = model or self.model
        try:
            for loaded in self.loaded_models():
                if loaded.name == name:
                    return loaded.context_length
        except LLMError:
            return 0
        return 0

    def refresh(self, model: str | None = None) -> ModelStatus:
        if model:
            self.model = model
        status = ModelStatus(current=self.model)
        try:
            status.installed = self.list_models()
            status.loaded = self.loaded_models()
            status.server_ok = True
        except LLMError as exc:
            status.error = str(exc)
            return status
        names = {m.name for m in status.installed}
        names |= {m.name.split(":")[0] for m in status.installed}
        if self.model not in names:
            status.error = (
                f"model '{self.model}' is not installed. "
                f"Available: {', '.join(m.name for m in status.installed) or 'none'}"
            )
        return status

    def require_model(self, model: str | None = None) -> str:
        name = model or self.model
        installed = [(m.name, m.name.split(":")[0]) for m in self.list_models()]
        flat = {part for pair in installed for part in pair}
        if name not in flat:
            raise LLMError(
                f"model '{name}' is not installed. Pull it with "
                f"`ollama pull {name}` or switch models."
            )
        return name

    def pull(self, model: str, *, timeout: int = 3600) -> str:
        try:
            proc = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError("ollama CLI not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"ollama pull timed out for {model}") from exc
        if proc.returncode != 0:
            raise LLMError((proc.stderr or proc.stdout or "").strip()[-1000:])
        return (proc.stdout or "").strip()

    # ---- generation ----------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        num_ctx: int | None = None,
        json_mode: bool = False,
        timeout: int | None = None,
        keep_alive: str = "10m",
        think: bool = False,
    ) -> str:
        if self._generate_override is not None:
            return self._generate_override(
                prompt, model=model or self.model, system=system,
                temperature=temperature, json_mode=json_mode,
            )
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "think": think,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        if json_mode:
            payload["format"] = "json"
        data = self._request(
            "/api/generate", payload, timeout=timeout or self.timeout,
        )
        body = data.get("response") or ""
        return strip_thinking(body)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: int | None = None,
        think: bool = False,
    ) -> str:
        return self.chat_detailed(
            messages,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
            timeout=timeout,
            think=think,
        )["content"]

    def chat_detailed(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: int | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Chat turn including Ollama's token usage for context displays."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "stream": False,
            "think": think,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        data = self._request("/api/chat", payload, timeout=timeout or self.timeout)
        message = data.get("message") or {}
        return {
            "content": strip_thinking(message.get("content") or ""),
            "model": data.get("model") or (model or self.model),
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "eval_tokens": int(data.get("eval_count") or 0),
            "done_reason": data.get("done_reason") or "",
        }

    # ---- embeddings ----------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]:
        inputs = [t for t in texts]
        if not inputs:
            return []
        name = model or self.embed_model
        try:
            data = self._request(
                "/api/embed",
                {"model": name, "input": inputs},
                timeout=timeout or DEFAULT_EMBED_TIMEOUT,
            )
            vectors = data.get("embeddings")
            if isinstance(vectors, list) and len(vectors) == len(inputs):
                return [[float(x) for x in vec] for vec in vectors]
            raise LLMError("Ollama returned malformed embeddings")
        except LLMError as exc:
            if "HTTP 404" not in str(exc):
                raise
        vectors = []
        for text in inputs:
            data = self._request(
                "/api/embeddings",
                {"model": name, "prompt": text},
                timeout=timeout or DEFAULT_EMBED_TIMEOUT,
            )
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise LLMError("Ollama returned malformed embedding")
            vectors.append([float(x) for x in vector])
        return vectors


def client_from_config(cfg: Config | None = None) -> OllamaClient:
    cfg = cfg or Config.load()
    return OllamaClient(
        base_url=cfg.ollama_url,
        model=cfg.model,
        embed_model=cfg.embed_model,
    )
