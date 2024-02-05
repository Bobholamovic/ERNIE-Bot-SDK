# Agents

## 1 智能体简介

在ERNIE Bot Agent框架中，智能体（agent）指的是可以通过行动自主完成预设目标的具有智能的实体。智能体具备自主理解、规划决策能力，能够执行复杂的任务。在用户与智能体的每一轮交互中，智能体接收一段自然语言文本作为输入，从输入中分析用户需求，确定需要完成的任务，然后通过调用外部工具等手段完成任务，并提供用户答复。ERNIE Bot Agent框架预置了一些智能体类，同时也支持开发者根据需要定制自己的智能体类。

## 2 使用预置智能体类

在阅读本节前，请首先熟悉chat models、tools和memory等模块的相关文档。此外，本节的所有示例代码均需要在异步环境中执行。例如，可以使用如下方式编写Python脚本执行示例代码：

```python
import asyncio

async def main():
    # 将示例代码拷贝到这里


if __name__ == "__main__":
    asyncio.run(main())
```

关于智能体相关类的详细接口，请参考[API文档](../package/erniebot_agent/agents.md)

#### 2.1 基础概念

为了更清晰地描述ERNIE Bot Agent框架中智能体的运行机制，在介绍具体的智能体类型之前，首先定义如下概念：

- **运行（run）**：指智能体与用户的一轮交互，包括从接收用户输入到给出答复的完整流程。
- **步骤（step）**：智能体在运行中执行一个行动并得到结果，称为执行一个步骤。一次运行通常包含一个或多个步骤。ERNIE Bot Agent框架提供以下几种内置步骤类型：

    - 工具步骤：调用智能体集成的一个工具。
    - 插件步骤：调用一个插件的某项功能。

#### 2.2 Function Agent

Function agent是一种由大语言模型的函数调用能力驱动的智能体，在ERNIE Bot Agent框架中对应`erniebot.agents.FunctionAgent`。

Function agent的一次运行可以划分为一系列迭代。在每次迭代中，由大语言模型决定智能体执行的步骤。具体而言，迭代开始时，function agent将当前对话上下文发送给模型。根据模型的回复消息，function agent作如下判断和处理：

- 如果模型仅给出自然语言文本，则将该文本作为本次运行提供给用户的回复文本，结束本次运行。
- 如果模型给出工具调用信息，则根据工具名称和参数找到并调用该工具，记录调用结果，完成一个工具步骤。之后，更新对话上下文，结束本次迭代。
- 如果模型在给出自然语言文本的同时也给出插件调用结果，则记录该结果，完成一个插件步骤。对于部分插件，在完成插件步骤后，智能体将不再具备执行其它步骤的能力，此时，以模型回复的自然语言文本为本次运行提供给用户的回复文本，结束本次运行；对于其它插件，则更新对话上下文，结束本次迭代。

以下是使用function agent的一些例子：

- 构造不配备有工具的function agent，用其进行多轮对话：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory

agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[], memory=WholeMemory())

response = await agent.run("你好，小度！")
# `text`属性是智能体给出的回复文本
print(response.text)
# 打印结果可能如下：
# 你好，有什么我可以帮你的吗？

response = await agent.run("我刚刚怎么称呼你？")
# `chat_history`属性存储本次运行中与模型的对话历史
for message in response.chat_history:
    print(message.content)
# 打印结果可能如下：
# 我刚刚怎么称呼你？
# 您叫我小度。如果您有任何问题或需要帮助，请随时告诉我。
```

- 使用function agent调用工具完成用户给定的任务：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.calculator_tool import CalculatorTool

# `Calculator`工具用于完成数学计算
# 如果没有传递`memory`参数，则`FunctionAgent`对象默认构造和使用一个`WholeMemory`对象
agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[CalculatorTool()])

response = await agent.run("请问四加5*捌的结果是多少？")

print(response.text)
# 打印结果可能如下：
# 根据您的公式4+5*8，结果是44。如果您还有其他问题或需要计算其他公式，请随时告诉我。

# `steps`属性存储本次运行的所有步骤
print(response.steps)
# 打印结果可能如下：
# [ToolStep(info={'tool_name': 'CalculatorTool', 'tool_args': '{"math_formula":"4+5*8"}'}, result='{"formula_result": 44}', input_files=[], output_files=[])]

step = response.steps[0]
print("调用的tool名称：", step.info["tool_name"])
print("调用tool输入的参数（JSON格式）：", step.info["tool_args"])
print("调用tool返回的结果（JSON格式）：", step.result)
```

- 使用function agent编排多工具：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.calculator_tool import CalculatorTool
from erniebot_agent.tools.current_time_tool import CurrentTimeTool

# 指定`enable_multi_step_tool_call`为True以启用多步工具调用功能
chat_model = ERNIEBot(model="ernie-3.5", enable_multi_step_tool_call=True)
agent = FunctionAgent(llm=chat_model, tools=[CalculatorTool()])

# 除了在构造智能体时指定工具，还可以通过`load_tool`方法加载工具
# `CurrentTimeTool`工具用于获取当前时间
agent.load_tool(CurrentTimeTool())
# 与`load_tool`相对，`unload_tool`方法可用于卸载工具

# `get_tools`方法返回智能体当前可以使用的所有工具
print(agent.get_tools())
# 打印结果如下：
# [<name: CalculatorTool, description: CalculatorTool用于执行数学公式计算>, <name: CurrentTimeTool, description: CurrentTimeTool 用于获取当前时间>]

response = await agent.run("请将当前时刻的时、分、秒数字相加，告诉我结果。")

print(response.text)
# 打印结果可能如下：
# 根据当前时间，时、分、秒数字相加的结果是68。

# 观察智能体是否正确调用工具完成任务
for step in response.steps:
    print(step)
# 打印结果可能如下：
# ToolStep(info={'tool_name': 'CurrentTimeTool', 'tool_args': '{}'}, result='{"current_time": "2023年12月27日 21时39分08秒"}', input_files=[], output_files=[])
# ToolStep(info={'tool_name': 'CalculatorTool', 'tool_args': '{"math_formula":"21+39+8"}'}, result='{"formula_result": 68}', input_files=[], output_files=[])
```

- 使用function agent调用输入、输出中包含文件的工具：

```python
import aiohttp

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.file import GlobalFileManagerHandler
from erniebot_agent.tools import RemoteToolkit

# 获取PP-OCRv4工具箱与语音合成工具箱，并将其中的所有工具装配给智能体
ocr_toolkit = RemoteToolkit.from_aistudio("pp-ocrv4")
tts_toolkit = RemoteToolkit.from_aistudio("texttospeech")
agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[*ocr_toolkit.get_tools(), *tts_toolkit.get_tools()])

# 下载示例图片
async with aiohttp.ClientSession() as session:
    async with session.get("https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/ocr_example_input.png") as response:
        with open("example.png", "wb") as f:
            f.write(await response.read())

# 获取文件管理器，并使用其创建文件对象
file_manager = GlobalFileManagerHandler().get()
input_file = await file_manager.create_file_from_path("example.png")

# 通过`files`参数在智能体运行输入中加入文件信息
response = await agent.run("请识别这张图片中的文字。", files=[input_file])

print(response.text)
# 打印结果可能如下：
# 根据您提供的图片，PP-OCRv4模型识别出了其中的文字，它们是“中国”和“汉字”。如果您需要更深入的分析或有其他问题，请随时告诉我。

assert len(response.steps) == 1
# `input_files`属性包含步骤涉及的所有输入文件
print(response.steps[0].input_files)
# 打印结果可能如下：
# [<LocalFile id: 'file-local-74aaf9e4-a4c2-11ee-b0a2-fa2020087eb4', filename: 'example.png', byte_size: 17663, created_at: '2023-12-27 22:15:58', purpose: 'assistants', metadata: {}, path: PosixPath('example.png')>]

# 尝试调用语音合成工具，该工具的输出中包含文件
response = await agent.run("请使用将刚才识别出的文字转换为语音。")

print(response.text)
# 打印结果可能如下：
# 根据您的需求，我为您合成了语音文件：file-local-4bab0eca-a4c3-11ee-a16e-fa2020087eb4。如果您需要进一步操作或有其他问题，请随时告诉我。

assert len(response.steps) == 1
# `output_files`属性包含步骤涉及的所有输出文件
output_files = response.steps[0].output_files
print(output_files)
# 打印结果可能如下：
# [<LocalFile id: 'file-local-4bab0eca-a4c3-11ee-a16e-fa2020087eb4', filename: 'tool-c3f5343c-6b33-4ee7-be31-35a60848bbd3.wav', byte_size: 43938, created_at: '2023-12-27 22:15:59', purpose: 'assistants_output', metadata: {'tool_name': 'texttospeech/v1.6/tts'}, path: PosixPath('/tmp/tmpd_ux8_ud/tool-c3f5343c-6b33-4ee7-be31-35a60848bbd3.wav')>]

# 将输出文件内容存储到指定文件中
assert len(output_files) == 1
await output_files[0].write_contents_to("output.wav")
```

#### 2.3 回调函数

为了使扩展智能体功能更加便利，ERNIE Bot Agent框架支持为`erniebot_agent.agents.Agent`的子类装配回调函数。具体而言，在初始化对象时可以传入`callbacks`参数，以使特定**事件**发生时相应的回调函数被调用。当未指定`callbacks`参数或者将其设置为`None`时，将使用默认的回调函数。

#### 2.3.1 事件一览

ERNIE Bot Agent框架定义了以下事件：

- `run_start`：智能体的运行开始。
- `llm_start`：智能体与模型的交互开始。
- `llm_end`：智能体与模型的交互成功结束。
- `llm_error`：智能体与模型的交互发生错误。
- `tool_start`：智能体对工具的调用开始。
- `tool_end`：智能体对工具的调用成功结束。
- `tool_error`：智能体对工具的调用发生错误。
- `run_error`：智能体的运行发生错误。
- `run_end`：智能体的运行成功结束。

#### 2.3.2 默认回调函数

默认装配的回调函数如下：

- `erniebot_agent.agents.callback.LoggingHandler`：日志记录回调函数集合。

#### 2.3.3 自定义回调函数

当默认回调函数无法满足需求时，可以通过继承基类`erniebot_agent.agents.callback.CallbackHandler`定制回调函数。具体而言，`erniebot_agent.agents.callback.CallbackHandler`提供一系列名为`on_{事件名称}`的方法，通过重写这些方法可以在特定事件发生时执行自定义逻辑。一个例子如下：

```python
from erniebot_agent.agents.callback import CallbackHandler

class CustomCallbackHandler(CallbackHandler):
    async def on_run_start(self, agent, prompt):
        print("智能体开始运行")

    async def on_run_end(self, agent, response):
        print("智能体结束运行，响应为：", response)
```

以上定义的`CustomCallbackHandler`在智能体开始运行和结束运行时打印信息。


## 3 定制智能体类

在部分情况下，预置的智能体类可能无法满足需求。为此，ERNIE Bot Agent框架也为用户提供定制智能体类的手段。在大部分情况下，推荐通过继承基类`erniebot_agent.agents.Agent`来实现这一目标。通常，`erniebot_agent.agents.Agent`的子类只需要重写`_run`方法，在其中实现自定义逻辑。

示例如下：

```python
from erniebot_agent.agents import Agent
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.memory.messages import HumanMessage

class CustomAgent(Agent):
    async def _run(self, prompt, files=None):
        # `chat_history`与`steps`分别用于记录本次运行的对话历史和步骤信息
        chat_history = []
        steps = []

        # 从`prompt`和`files`构建输入消息
        prompt_with_file_reprs = await self.add_file_reprs_to_text(prompt, files)
        input_message = HumanMessage(content=prompt_with_file_reprs)
        chat_history.append(input_message)

        # 与模型交互
        llm_resp = await self.run_llm(self.memory.get_messages())
        chat_history.append(llm_resp.message)

        # 根据模型的输出，决定接下来的步骤
        ...

        # 假设用`should_run_tool`指示是否应该调用工具，在`action`中包含工具名称和输入参数
        if should_run_tool:
            # 调用工具
            tool_resp = await self.run_tool(action["tool_name"], action["tool_args"])
            # 将`tool_resp`转换为`erniebot_agent.agents.schema.ToolStep`对象`tool_step`
            ...
            steps.append(tool_step)
            # 假设自定义的智能体在此处准备结束运行
            # 更新记忆
            # 出于精简对话历史的考虑，这里仅将本次运行的第一条和最后一条消息加入智能体记忆中
            self.memory.add_message(chat_history[0])
            self.memory.add_message(chat_history[-1])
            # 构造`AgentResponse`对象并返回
            # 智能体已经完成任务，所以将`end_reason`设置为"FINISHED"
            return AgentResponse(
                text=chat_history[-1].content,
                chat_history=chat_history,
                steps=steps,
                end_reason="FINISHED",
            )

        # 其它处理逻辑
        ...
```