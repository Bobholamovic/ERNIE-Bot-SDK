from typing import Generator

import pytest
from langchain_core.outputs import LLMResult

from erniebot_agent.extensions.langchain.llms import ERNIE


def test_ernie_call() -> None:
    """Test valid call."""
    llm = ERNIE()
    output = llm("Hi, ernie.")
    assert isinstance(output, str)


def test_ernie_generate() -> None:
    """Test generation."""
    llm = ERNIE()
    output = llm.generate(["Hi, ernie."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


@pytest.mark.asyncio
async def test_ernie_agenerate() -> None:
    """Test asynchronous generation."""
    llm = ERNIE()
    output = await llm.agenerate(["Hi, ernie."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_ernie_streaming_generate() -> None:
    """Test generation with streaming enabled."""
    llm = ERNIE(streaming=True)
    output = llm.generate(["Write a joke."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_ernie_stream() -> None:
    """Test streaming."""
    llm = ERNIE()
    output = llm.stream("Write a joke.")
    assert isinstance(output, Generator)
    for res in output:
        assert isinstance(res, str)


@pytest.mark.asyncio
async def test_ernie_astream() -> None:
    """Test asynchronous streaming."""
    llm = ERNIE()
    output = llm.astream("Write a joke.")
    async for res in output:
        assert isinstance(res, str)
