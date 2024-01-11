import pytest

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.memory.messages import SystemMessage


@pytest.fixture
def erniebot_with_aistudio_backend_with_atoken():
    erniebot = ERNIEBot(
        model="ernie-3.5",
        api_type="aistudio",
        access_token="access_token",
        enable_multi_step_tool_call=False,
        enable_human_clarify=False,
    )
    return erniebot


@pytest.fixture
def erniebot_with_aistudio_backend_wo_atoken():
    erniebot = ERNIEBot(
        model="ernie-3.5",
        api_type="aistudio",
        access_token=None,
        enable_multi_step_tool_call=False,
        enable_human_clarify=False,
    )
    return erniebot


@pytest.fixture
def erniebot_with_qianfan_backend_aksk():
    erniebot = ERNIEBot(
        model="ernie-3.5",
        api_type="qianfan",
        access_token="access_token",
        ak="ak",
        sk="sk",
        enable_multi_step_tool_call=False,
        enable_human_clarify=False,
    )
    return erniebot


@pytest.fixture
def erniebot_with_qianfan_no_access_token_backend_aksk():
    erniebot = ERNIEBot(
        model="ernie-3.5",
        api_type="qianfan",
        access_token=None,
        ak="ak",
        sk="sk",
        enable_multi_step_tool_call=False,
        enable_human_clarify=False,
    )
    return erniebot


@pytest.fixture
def erniebot_with_qianfan_no_access_token_backend():
    erniebot = ERNIEBot(
        model="ernie-3.5",
        api_type="qianfan",
        access_token="access_token",
        enable_multi_step_tool_call=False,
        enable_human_clarify=False,
    )
    return erniebot


@pytest.mark.asyncio
async def test_erniebot_aistudio_backend_with_atoken_base(erniebot_with_aistudio_backend_with_atoken):
    erniebot = erniebot_with_aistudio_backend_with_atoken
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=["eChart"],
    )
    assert cfg_dict["_config_"]["api_type"] == "aistudio"
    assert ("ak" in cfg_dict["_config_"]) is False
    assert ("sk" in cfg_dict["_config_"]) is False
    assert cfg_dict["_config_"]["access_token"] == "access_token"
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["plugins"] == ["eChart"]
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]


@pytest.mark.asyncio
async def test_erniebot_aistudio_backend_with_atoken_no_plugin(
    erniebot_with_aistudio_backend_with_atoken,
):
    erniebot = erniebot_with_aistudio_backend_with_atoken
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=[],
    )
    assert cfg_dict["_config_"]["api_type"] == "aistudio"
    assert ("ak" in cfg_dict["_config_"]) is False
    assert ("sk" in cfg_dict["_config_"]) is False
    assert cfg_dict["_config_"]["access_token"] == "access_token"
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]
    assert ("plugins" in cfg_dict) is False


@pytest.mark.asyncio
async def test_erniebot_aistudio_backend_wo_atoken_base(erniebot_with_aistudio_backend_wo_atoken):
    erniebot = erniebot_with_aistudio_backend_wo_atoken
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=[],
    )
    assert cfg_dict["_config_"]["api_type"] == "aistudio"
    assert ("ak" in cfg_dict["_config_"]) is False
    assert ("sk" in cfg_dict["_config_"]) is False
    assert cfg_dict["_config_"]["access_token"] is None
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]
    assert ("plugins" in cfg_dict) is False


@pytest.mark.asyncio
async def test_erniebot_qianfan_backend_with_atoken_base(erniebot_with_qianfan_backend_aksk):
    erniebot = erniebot_with_qianfan_backend_aksk
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=[],
    )
    assert cfg_dict["_config_"]["api_type"] == "qianfan"
    assert cfg_dict["_config_"]["ak"] == "ak"
    assert cfg_dict["_config_"]["sk"] == "sk"
    assert cfg_dict["_config_"]["access_token"] == "access_token"
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]
    assert ("plugins" in cfg_dict) is False


@pytest.mark.asyncio
async def test_erniebot_qianfan_backend_wo_atoken_base(
    erniebot_with_qianfan_no_access_token_backend_aksk,
):
    erniebot = erniebot_with_qianfan_no_access_token_backend_aksk
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=[],
    )
    assert cfg_dict["_config_"]["api_type"] == "qianfan"
    assert cfg_dict["_config_"]["ak"] == "ak"
    assert cfg_dict["_config_"]["sk"] == "sk"
    assert cfg_dict["_config_"]["access_token"] is None
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]
    assert ("plugins" in cfg_dict) is False


@pytest.mark.asyncio
async def test_erniebot_qianfan_backend_with_atoken_noaksk(
    erniebot_with_qianfan_no_access_token_backend,
):
    erniebot = erniebot_with_qianfan_no_access_token_backend
    messages = [HumanMessage("今天深圳天气怎么样？")]

    cfg_dict = erniebot._generate_config(
        messages,
        functions=[{}],
        top_p=5,
        temperature=0.8,
        penalty_score=10,
        system=SystemMessage("这是一条系统信息"),
        plugins=[],
    )
    assert cfg_dict["_config_"]["api_type"] == "qianfan"
    assert ("ak" in cfg_dict["_config_"]) is False
    assert ("sk" in cfg_dict["_config_"]) is False
    assert cfg_dict["_config_"]["access_token"] == "access_token"
    assert cfg_dict["model"] == "ernie-3.5"
    assert cfg_dict["top_p"] == 5
    assert cfg_dict["temperature"] == 0.8
    assert cfg_dict["penalty_score"] == 10
    assert cfg_dict["system"].to_dict() == SystemMessage("这是一条系统信息").to_dict()
    assert cfg_dict["messages"] == [HumanMessage("今天深圳天气怎么样？").to_dict()]
    assert ("plugins" in cfg_dict) is False
