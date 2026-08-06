"""Unit tests for the generator (LLM) layer and prompt building."""

import asyncio

import pytest

from app.generator import OpenAIGenerator, SynthesisGenerator
from app.rag import RagEngine


def asyncio_run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_build_prompt_includes_question_and_sources():
    system, user = RagEngine.build_prompt(
        "What is RAG?",
        [{"doc_id": "a.md", "text": "RAG means retrieval-augmented generation."}],
    )
    assert "docchat" in system
    assert "What is RAG?" in user
    assert "a.md" in user


def test_synthesis_generator_yields_top_chunk(monkeypatch):
    rag = RagEngine()
    rag.ingest_texts(1, {"payments.md": "Settlement moves money to a merchant bank."})
    rag.ask(1, "How does money reach a merchant?", k=1)

    gen = SynthesisGenerator(rag)

    async def collect():
        return [tok async for tok in gen.stream("sys", "user")]

    tokens = asyncio_run(collect())
    joined = "".join(tokens)
    assert "payments.md" in joined
    assert "merchant" in joined


def test_openai_generator_requires_key(monkeypatch):
    monkeypatch.setattr("app.generator.settings.openai_api_key", "")
    with pytest.raises(ValueError):
        OpenAIGenerator(api_key="")
