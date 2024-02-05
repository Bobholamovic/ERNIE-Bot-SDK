from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory import AIMessage


class FakeSimpleChatModel(ChatModel):
    def __init__(self):
        super().__init__("simple_chat_model")

    @property
    def response(self):
        return AIMessage(content="Text response", function_call=None, token_usage=None)

    async def chat(
        self,
        messages,
        *,
        stream=False,
        functions=None,
        system=None,
        plugins=None,
        tool_choice=None,
        **kwargs,
    ):
        if stream:
            raise ValueError("Streaming is not supported.")
        if system is not None:
            response = f"Recieved system message: {system}"
            return AIMessage(content=response, function_call=None, token_usage=None)
        # Ignore other arguments
        return self.response


class FakeChatModelWithPresetResponses(ChatModel):
    def __init__(self, responses):
        super().__init__("erniebot_with_preset_responses")
        self.responses = responses
        self._counter = 0

    async def chat(
        self,
        messages,
        *,
        stream=False,
        functions=None,
        system=None,
        plugins=None,
        tool_choice=None,
        **kwargs,
    ):
        if stream:
            raise ValueError("Streaming is not supported.")
        # Ignore other arguments
        response = self.responses[self._counter]
        self._counter += 1
        return response
