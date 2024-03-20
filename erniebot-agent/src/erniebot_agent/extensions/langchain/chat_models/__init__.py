import warnings
from typing import Any

from erniebot_agent.extensions.langchain.chat_models.ernie import ChatERNIE


def __getattr__(name: str) -> Any:
    if name == "ErnieBotChat":
        warnings.warn("`ErnieBotChat` is deprecated. Please use `ChatERNIE` instead.", FutureWarning)
        return ChatERNIE
    else:
        raise AttributeError(f"module {repr(__name__)} has no attribute {repr(name)}")
