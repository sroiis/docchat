"""The GENERATION half of RAG: turning retrieved chunks into a fluent answer.

This is the seam the original docchat deliberately left out. It supports:

  none    -> retrieval-only: no LLM call, returns a synthesis of the top chunk
  ollama  -> a local LLM (fully offline, no API keys)
  openai  -> any OpenAI-compatible /chat/completions API

The `Generator` protocol exposes a single async streaming method, so the HTTP
layer can push tokens to the browser as they arrive.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from .config import settings


class Generator(Protocol):
    name: str

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they are generated."""
        ...


class SynthesisGenerator:
    """Retrieval-only fallback: emit the best chunk's text as the "answer"."""

    name = "none"

    def __init__(self, rag) -> None:
        self._rag = rag  # RagEngine, used to synthesise from the top hit

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        # The first retrieved chunk carries the synthesized answer line, then
        # the raw text. The rag layer guarantees at least one source here.
        best = self._rag.last_answer
        if best and best.sources:
            src = best.sources[0]
            yield f"Based on '{src['doc_id']}' (relevance {src['score']:.0%}):\n\n"
            yield src["text"]


class OllamaGenerator:
    """Stream from a local Ollama server (`ollama serve`)."""

    name = "ollama"

    def __init__(
        self,
        model: str = settings.llm_model,
        base_url: str = settings.ollama_base_url,
        temperature: float = settings.llm_temperature,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {"temperature": self.temperature},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = payload.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if payload.get("done", False):
                        break


class OpenAIGenerator:
    """Stream from any OpenAI-compatible `/chat/completions` endpoint."""

    name = "openai"

    def __init__(
        self,
        model: str = settings.llm_model,
        base_url: str = settings.openai_base_url,
        api_key: str = settings.openai_api_key,
        temperature: float = settings.llm_temperature,
    ):
        if not api_key:
            raise ValueError("DOCCHAT_OPENAI_API_KEY is required for openai generation")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = payload.get("choices") or []
                    delta = choices[0].get("delta", {}) if choices else {}
                    piece = delta.get("content", "")
                    if piece:
                        yield piece


def build_generator(rag) -> Generator:
    """Instantiate the generator selected by DOCCHAT_LLM_PROVIDER.

    `rag` is passed so the 'none' provider can synthesise an answer from the
    just-retrieved top chunk.
    """
    provider = settings.llm_provider.lower()
    if provider == "none":
        return SynthesisGenerator(rag)
    if provider == "ollama":
        return OllamaGenerator()
    if provider == "openai":
        return OpenAIGenerator()
    raise ValueError(
        f"Unknown llm_provider {provider!r}; choose none | ollama | openai"
    )
