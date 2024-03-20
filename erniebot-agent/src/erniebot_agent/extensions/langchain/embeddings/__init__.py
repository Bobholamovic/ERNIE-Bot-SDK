import warnings
from typing import Any

from erniebot_agent.extensions.langchain.embeddings.ernie import ERNIEEmbeddings


def __getattr__(name: str) -> Any:
    if name == "ErnieEmbeddings":
        warnings.warn(
            "`ErnieEmbeddings` is deprecated. Please use `ERNIEEmbeddings` instead.", FutureWarning
        )
        return ERNIEEmbeddings
    else:
        raise AttributeError(f"module {repr(__name__)} has no attribute {repr(name)}")
