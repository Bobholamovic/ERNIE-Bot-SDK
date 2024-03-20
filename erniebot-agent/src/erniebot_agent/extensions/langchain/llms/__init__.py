import warnings
from typing import Any

from erniebot_agent.extensions.langchain.llms.ernie import ERNIE


def __getattr__(name: str) -> Any:
    if name == "ErnieBot":
        warnings.warn("`ErnieBot` is deprecated. Please use `ERNIE` instead.", FutureWarning)
        return ERNIE
    else:
        raise AttributeError(f"module {repr(__name__)} has no attribute {repr(name)}")
