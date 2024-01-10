# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod
from typing import Any, AsyncIterator, List, Literal, Optional, Union, overload

from erniebot_agent.memory.messages import AIMessage, AIMessageChunk, Message


class ChatModel(metaclass=ABCMeta):
    """The base class of chat-optimized LLM.

    Attributes:
        model (str): The model name.
        default_chat_kwargs (Any): A dict for setting default args for chat model,
            the supported keys include `model`, `_config_`, `top_p`, etc.
    """

    def __init__(self, model: str, **default_chat_kwargs: Any):
        self.model = model
        self.default_chat_kwargs = default_chat_kwargs

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[False] = ...,
        functions: Optional[List[dict]] = ...,
        system: Optional[str] = ...,
        plugins: Optional[List[str]] = ...,
        tool_choice: Optional[dict] = ...,
        **kwargs: Any,
    ) -> AIMessage:
        ...

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: Literal[True],
        functions: Optional[List[dict]] = ...,
        system: Optional[str] = ...,
        plugins: Optional[List[str]] = ...,
        tool_choice: Optional[dict] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        ...

    @overload
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: bool,
        functions: Optional[List[dict]] = ...,
        system: Optional[str] = ...,
        plugins: Optional[List[str]] = ...,
        tool_choice: Optional[dict] = ...,
        **kwargs: Any,
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        functions: Optional[List[dict]] = None,
        system: Optional[str] = None,
        plugins: Optional[List[str]] = None,
        tool_choice: Optional[dict] = None,
        **kwargs: Any,
    ) -> Union[AIMessage, AsyncIterator[AIMessageChunk]]:
        """The abstract method for asynchronously chatting with the LLM.

        Args:
            messages (List[Message]): A list of messages.
            stream (bool): Whether to use streaming generation. Defaults to
                False.
            functions (Optional[List[dict]]): The function descriptions to be
                used by the model. Defaults to None.
            system (Optional[str]): The system message to be used by the model.
                Defaults to None.
            plugins (Optional[List[str]]): The names of the plugins to be used
                by the model. Defaults to None.
            tool_choice (Optional[dict]): The information about the tool to be
                chosen by the model. Defaults to None.
            **kwargs: Keyword arguments, such as `top_p`, `temperature`, and
                `penalty_score`.

        Returns:
            If `stream` is False, returns a single message.
            If `stream` is True, returns an asynchronous iterator of message
            chunks.
        """
        raise NotImplementedError
