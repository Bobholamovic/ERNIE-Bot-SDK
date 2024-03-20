import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from erniebot_agent.extensions.langchain.chat_models import ChatERNIE


def test_ernie_call() -> None:
    """Test valid call."""
    chat = ChatERNIE()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_ernie_generate() -> None:
    """Test generation."""
    chat = ChatERNIE()
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_ernie_agenerate() -> None:
    """Test asynchronous generation."""
    chat = ChatERNIE()
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_ernie_stream() -> None:
    """Test streaming."""
    chat = ChatERNIE()
    for chunk in chat.stream("Write a joke."):
        assert isinstance(chunk.content, str)


@pytest.mark.asyncio
async def test_ernie_astream() -> None:
    """Test asynchronous streaming."""
    chat = ChatERNIE()
    async for chunk in chat.astream("Write a joke."):
        assert isinstance(chunk.content, str)


def test_ernie_params() -> None:
    """Test setting parameters."""
    chat = ChatERNIE(model="ernie-turbo", temperature=0.7)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_ernie_chat_history() -> None:
    """Test that multiple messages works."""
    chat = ChatERNIE()
    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
