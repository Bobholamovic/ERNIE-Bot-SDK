import abc
import json
import logging
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    final,
)

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.mixins import GradioMixin
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file import (
    File,
    FileManager,
    GlobalFileManagerHandler,
    get_default_file_manager,
    protocol,
)
from erniebot_agent.memory import Memory, WholeMemory
from erniebot_agent.memory.messages import Message, SystemMessage
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager

_PLUGINS_WO_FILE_IO: Final[Tuple[str]] = ("eChart",)

_logger = logging.getLogger(__name__)


class Agent(GradioMixin, BaseAgent):
    """The base class for agents.

    Typically, this class should be the base class for custom agent classes. A
    class derived from this class must implement how the agent orchestates the
    components to complete tasks.

    Attributes:
        llm: The LLM that the agent uses.
        memory: The message storage that keeps the chat history.
    """

    llm: ChatModel
    memory: Memory

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
    ) -> None:
        """Initialize an agent.

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
        """
        super().__init__()

        self.llm = llm

        if isinstance(tools, ToolManager):
            self._tool_manager = tools
        else:
            self._tool_manager = ToolManager(tools)

        if memory is None:
            memory = self._create_default_memory()
        self.memory = memory

        self.system = SystemMessage(system) if system is not None else system
        if self.system is not None:
            self.memory.set_system_message(self.system)

        if callbacks is None:
            callbacks = get_default_callbacks()
        if isinstance(callbacks, CallbackManager):
            self._callback_manager = callbacks
        else:
            self._callback_manager = CallbackManager(callbacks)

        self._file_manager = file_manager or get_default_file_manager()

        self._plugins: List[str] = list(plugins) if plugins is not None else []
        self._init_file_needs_url()

        self._is_running = False

    @final
    async def run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        """Run the agent asynchronously.

        Args:
            prompt: A natural language text describing the task that the agent
                should perform.
            files: A sequence of files that the agent can use to perform the
                task.

        Returns:
            The agent's response, containing the final answer and intemediate
            data.
        """
        if self._is_running:
            raise RuntimeError("The agent is already running.")
        self._is_running = True
        await self._callback_manager.on_run_start(agent=self, prompt=prompt, files=files)
        try:
            agent_resp = await self._run(prompt, files)
        except BaseException as e:
            await self._callback_manager.on_run_error(agent=self, error=e)
            raise e
        else:
            await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        finally:
            self._is_running = False
        return agent_resp

    @final
    async def run_llm(
        self,
        messages: List[Message],
        *,
        functions: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        llm_opts: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        """Run the LLM asynchronously.

        Args:
            messages: The input messages.
            functions: The function descriptions to be used by the LLM.
            tool_choice: The name of the tool to be chosen by the LLM.
            llm_opts: The additional options to pass to the LLM.

        Returns:
            The LLM's response.
        """
        await self._callback_manager.on_llm_start(agent=self, llm=self.llm, messages=messages)
        try:
            llm_resp = await self._run_llm(
                messages,
                functions=functions,
                tool_choice=tool_choice,
                llm_opts=llm_opts,
            )
        except BaseException as e:
            await self._callback_manager.on_llm_error(agent=self, llm=self.llm, error=e)
            raise
        else:
            await self._callback_manager.on_llm_end(agent=self, llm=self.llm, response=llm_resp)
        return llm_resp

    @final
    async def run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        """Call the specified tool asynchronously.

        Args:
            tool_name: The name of the tool to call.
            tool_args: The tool arguments in JSON format.

        Returns:
            The tool's response, containing the result of the tool call.
        """
        tool = self._tool_manager.get_tool(tool_name)
        await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        try:
            tool_resp = await self._run_tool(tool, tool_args)
        except BaseException as e:
            await self._callback_manager.on_tool_error(agent=self, tool=tool, error=e)
            raise e
        else:
            await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    def load_tool(self, tool: BaseTool) -> None:
        """Load a tool into the agent.

        Args:
            tool: The tool to load.
        """
        self._tool_manager.add_tool(tool)

    def unload_tool(self, tool: Union[BaseTool, str]) -> None:
        """Unload a tool from the agent.

        Args:
            tool: The tool to unload.
        """
        if isinstance(tool, str):
            tool = self.get_tool(tool)
        self._tool_manager.remove_tool(tool)

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a tool by name."""
        return self._tool_manager.get_tool(tool_name)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools that the agent can choose from."""
        return self._tool_manager.get_tools()

    def reset_memory(self) -> None:
        """Clear the chat history."""
        self.memory.clear_chat_history()

    def get_file_manager(self) -> FileManager:
        # Can we create a lazy proxy for the global file manager and simply set
        # and use `self._file_manager`?
        if self._file_manager is None:
            file_manager = GlobalFileManagerHandler().get()
        else:
            file_manager = self._file_manager
        return file_manager

    async def add_file_reprs_to_text(self, text: str, files: Sequence[File]) -> str:
        def _append_files_repr_to_text(text: str, files_repr: str) -> str:
            return f"{text}\n{files_repr}"

        if len(files) > 0:
            if len(protocol.extract_file_ids(text)) > 0:
                _logger.warning("File IDs were found in the text. The provided files will be ignored.")
            else:
                file_manager = self.get_file_manager()
                file_reprs = await file_manager.create_file_reprs(files, include_urls=self.file_needs_url)
                files_repr = "\n".join(file_reprs)
                text = _append_files_repr_to_text(text, files_repr)
        return text

    @abc.abstractmethod
    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        raise NotImplementedError

    async def _run_llm(
        self,
        messages: List[Message],
        *,
        functions: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        llm_opts: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        if functions is None:
            functions = self._tool_manager.get_tool_schemas()
        tool_choice_for_llm: Optional[dict] = None
        if tool_choice is not None:
            tool_choice_for_llm = {"type": "function", "function": {"name": tool_choice}}
        system = self.system.content if self.system is not None else None
        plugins = self._plugins
        llm_ret = await self.llm.chat(
            messages,
            stream=False,
            functions=functions,
            system=system,
            plugins=plugins,
            tool_choice=tool_choice_for_llm,
            **(llm_opts or {}),
        )
        return LLMResponse(message=llm_ret)

    async def _run_tool(self, tool: BaseTool, tool_args: str) -> ToolResponse:
        parsed_tool_args = self._parse_tool_args(tool_args)
        file_manager = self.get_file_manager()
        # XXX: Sniffing is error-prone and less efficient. Further, the result
        # can be incorrect. Can we make a protocol to statically recognize file
        # inputs and outputs or can we have the tools introspect about this?
        input_files = await file_manager.sniff_and_extract_files_from_obj(parsed_tool_args)
        tool_ret = await tool(**parsed_tool_args)
        if isinstance(tool_ret, dict):
            output_files = await file_manager.sniff_and_extract_files_from_obj(tool_ret)
        else:
            output_files = []
        tool_ret_json = json.dumps(tool_ret, ensure_ascii=False)
        return ToolResponse(json=tool_ret_json, input_files=input_files, output_files=output_files)

    def _create_default_memory(self) -> Memory:
        return WholeMemory()

    def _init_file_needs_url(self):
        self.file_needs_url = False
        for plugin in self._plugins:
            if plugin not in _PLUGINS_WO_FILE_IO:
                self.file_needs_url = True
                break

    def _parse_tool_args(self, tool_args: str) -> Dict[str, Any]:
        try:
            args_dict = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ValueError(f"`tool_args` cannot be parsed as JSON. `tool_args`: {tool_args}")

        if not isinstance(args_dict, dict):
            raise ValueError(f"`tool_args` cannot be interpreted as a dict. `tool_args`: {tool_args}")
        return args_dict
