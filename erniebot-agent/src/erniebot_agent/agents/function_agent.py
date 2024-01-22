# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import (
    AgentResponse,
    AgentRunEnd,
    AgentStep,
    PluginStep,
    ToolInfo,
    ToolStep,
)
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file import File, FileManager
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    Message,
)
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager

_logger = logging.getLogger(__name__)


class FunctionAgent(Agent):
    """An agent driven by function calling.

    The orchestration capabilities of a function agent are powered by the
    function calling ability of LLMs. A typical run of a function agent consists
    of one or more iterations. In each iteration, the LLM is asked to decide the
    next step the agent should take. The agent run does not end until the
    maximum number of iterations is reached or the LLM considers the task
    finished.

    Attributes:
        llm: The LLM that the agent uses.
        memory: The message storage that keeps the chat history.
        max_num_iters: The maximum iterations in each agent run.
    """

    llm: ChatModel
    memory: Memory
    max_num_iters: int

    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, Iterable[BaseTool]],
        *,
        memory: Optional[Memory] = None,
        system: Optional[str] = None,
        callbacks: Optional[Union[CallbackManager, Iterable[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[Iterable[str]] = None,
        max_num_iters: int = 5,
        first_tools: Optional[Sequence[BaseTool]] = None,
    ) -> None:
        """Initialize a function agent.

        Args:
            llm: An LLM for the agent to use.
            tools: The tools for the agent to use.
            memory: A memory object that equips the agent to remember chat
                history. If not specified, a new WholeMemory object will be
                instantiated.
            system: A message that tells the LLM how to interpret the
                conversations.
            callbacks: A list of callback handlers for the agent to use. If
                `None`, a default list of callbacks will be used.
            file_manager: A file manager for the agent to interact with files.
                If `None`, a global file manager that can be shared among
                different components will be implicitly created and used.
            plugins: The names of the plugins for the agent to use. If `None`,
                the agent will use a default list of plugins. Set `plugins` to
                `[]` to disable the use of plugins.
            max_num_iters: The maximum number of iterations in each agent run.
            first_tools: The tools arranged to be called sequentially before the
                agent run iterations.

        Raises:
            ValueError: If `max_num_iters` is not positive.
            RuntimeError: If tools in `first_tools` do not exist in `tools`.
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system=system,
            callbacks=callbacks,
            file_manager=file_manager,
            plugins=plugins,
        )

        if max_num_iters <= 0:
            raise ValueError("Invalid `max_num_iters` value")
        self.max_num_iters = max_num_iters

        if first_tools:
            for tool in first_tools:
                if tool not in self.get_tools():
                    raise ValueError("The tool in `first_tools` must be in the tools list.")
            self._first_tools = first_tools
        else:
            self._first_tools = []

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        chat_history, steps = await self._initialize_run(prompt, files)

        for tool in self._first_tools:
            new_steps, new_messages = await self._take_next_steps(chat_history, -1, selected_tool=tool)
            if len(new_steps) == 1 and isinstance(new_steps[0], ToolStep):
                chat_history.extend(new_messages)
                steps.append(new_steps[0])
            else:
                # If tool choice did not work, issue a warning and try next one.
                _logger.warning("The selected tool %r was not called.", tool.tool_name)

        curr_iter = 0
        while True:
            new_steps, new_messages = await self._take_next_steps(chat_history, curr_iter)
            chat_history.extend(new_messages)
            for step in new_steps:
                if isinstance(step, AgentRunEnd):
                    # The `AgentStep` objects that appear after an `AgentRunEnd`
                    # object will not be recorded.
                    # Should we perform checks and issue a warning here?
                    return await self._finalize_run(step, chat_history, steps)
                else:
                    steps.append(step)
            curr_iter += 1

    async def _initialize_run(
        self, prompt: str, files: Optional[Sequence[File]]
    ) -> Tuple[List[Message], List[AgentStep]]:
        chat_history: List[Message] = []
        steps: List[AgentStep] = []

        if files is not None:
            prompt_with_file_reprs = await self.add_file_reprs_to_text(prompt, files)
            run_input = HumanMessage(content=prompt_with_file_reprs)
        else:
            run_input = HumanMessage(content=prompt)

        chat_history.append(run_input)

        return chat_history, steps

    async def _take_next_steps(
        self,
        chat_history: List[Message],
        curr_iter: int,
        selected_tool: Optional[BaseTool] = None,
    ) -> Tuple[List[Union[AgentStep, AgentRunEnd]], List[Message]]:
        if not chat_history:
            raise ValueError("`chat_history` should not be empty.")
        if isinstance(chat_history[-1], AIMessage):
            raise ValueError("The last element in `chat_history` should not be an `AIMessage` object.")

        if curr_iter >= self.max_num_iters:
            response_text = "The agent run was stopped because the maximum number of iterations was reached."
            return [
                AgentRunEnd(
                    response=response_text,
                    end_reason="STOPPED",
                )
            ], []

        input_messages = self.memory.get_messages() + chat_history

        if selected_tool is not None:
            llm_resp = await self.run_llm(
                messages=input_messages,
                functions=[selected_tool.function_call_schema()],  # Only provide the selected tool
                tool_choice=selected_tool.tool_name,
            )
        else:
            llm_resp = await self.run_llm(messages=input_messages)

        new_steps: List[Union[AgentStep, AgentRunEnd]] = []
        new_messages: List[Message] = []
        output_message = llm_resp.message  # AIMessage
        new_messages.append(output_message)
        if output_message.function_call is not None:
            tool_name = output_message.function_call["name"]
            tool_args = output_message.function_call["arguments"]
            tool_resp = await self.run_tool(tool_name=tool_name, tool_args=tool_args)
            new_messages.append(FunctionMessage(name=tool_name, content=tool_resp.json))
            new_steps.append(
                ToolStep(
                    info=ToolInfo(tool_name=tool_name, tool_args=tool_args),
                    result=tool_resp.json,
                    input_files=tool_resp.input_files,
                    output_files=tool_resp.output_files,
                )
            )
        elif output_message.plugin_info is not None:
            file_manager = self.get_file_manager()
            new_steps.append(
                PluginStep(
                    info=output_message.plugin_info,
                    result=output_message.content,
                    input_files=await file_manager.sniff_and_extract_files_from_text(
                        input_messages[-1].content
                    ),  # TODO: make sure this is correct.
                    output_files=[],
                )
            )
            new_steps.append(AgentRunEnd(response=output_message.content, end_reason="FINISHED"))
        else:
            if output_message.clarify:
                # `clarify` and [`function_call`, `plugin`(directly end)] will not appear at the same time.
                new_steps.append(AgentRunEnd(response=output_message.content, end_reason="CLARIFY"))
            else:
                new_steps.append(AgentRunEnd(response=output_message.content, end_reason="FINISHED"))
        return new_steps, new_messages

    async def _finalize_run(
        self,
        end_info: AgentRunEnd,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        if not chat_history:
            raise ValueError("`chat_history` should not be empty.")

        response = AgentResponse(
            text=end_info.response,
            chat_history=chat_history,
            steps=steps,
            end_reason=end_info.end_reason,
        )

        if response.end_reason != "STOPPED":
            first_message = chat_history[0]
            last_message = chat_history[-1]
            if not (
                len(chat_history) >= 2
                and isinstance(first_message, HumanMessage)
                and isinstance(last_message, AIMessage)
            ):
                raise ValueError("The chat history is incomplete or invalid.")
            self.memory.add_message(first_message)
            self.memory.add_message(last_message)

        return response
