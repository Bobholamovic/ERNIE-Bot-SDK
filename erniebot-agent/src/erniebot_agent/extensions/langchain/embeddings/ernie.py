from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class ERNIEEmbeddings(BaseModel, Embeddings):
    """ERNIE embedding models.

    To use, you should have the ``erniebot`` python package installed, and the
    environment variable ``AISTUDIO_ACCESS_TOKEN`` set with your AI Studio
    access token.

    Example:
        .. code-block:: python
            from erniebot_agent.extensions.langchain.embeddings import ERNIEEmbeddings
            ernie_embeddings = ERNIEEmbeddings()
    """

    client: Any = None
    chunk_size: int = 16
    """Chunk size to use when the input is a list of texts."""
    aistudio_access_token: Optional[SecretStr] = None
    """AI Studio access token."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    model: str = "ernie-text-embedding"
    """Model to use."""
    request_timeout: Optional[int] = 60
    """How many seconds to wait for the server to send data before giving up."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        aistudio_access_token = get_from_dict_or_env(
            values,
            "aistudio_access_token",
            "AISTUDIO_ACCESS_TOKEN",
        )
        values["aistudio_access_token"] = convert_to_secret_str(aistudio_access_token)

        try:
            import erniebot

            values["client"] = erniebot.Embedding
        except ImportError:
            raise ImportError(
                "Could not import erniebot python package. Please install it with `pip install erniebot`."
            )
        return values

    def embed_query(self, text: str) -> List[float]:
        resp = self.embed_documents([text])
        return resp[0]

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [texts[i : i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]
        lst = []
        for chunk in text_in_chunks:
            resp = self.client.create(
                _config_={"max_retries": self.max_retries, **self._get_auth_config()},
                input=chunk,
                model=self.model,
            )
            lst.extend([res["embedding"] for res in resp["data"]])
        return lst

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [texts[i : i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]
        lst = []
        for chunk in text_in_chunks:
            resp = await self.client.acreate(
                _config_={"max_retries": self.max_retries, **self._get_auth_config()},
                input=chunk,
                model=self.model,
            )
            for res in resp["data"]:
                lst.extend([res["embedding"]])
        return lst

    def _get_auth_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {"api_type": "aistudio"}
        if self.aistudio_access_token:
            config["access_token"] = self.aistudio_access_token.get_secret_value()
        return config
