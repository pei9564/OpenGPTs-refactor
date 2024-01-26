from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    ConfigurableField,
    ConfigurableFieldMultiOption,
    RunnableBinding,
)

from app.agent_types.google_agent import get_google_agent_executor
from app.agent_types.openai_agent import get_openai_agent_executor
from app.agent_types.xml_agent import get_xml_agent_executor
from app.checkpoint import RedisCheckpoint
from app.llms import get_anthropic_llm, get_google_llm, get_openai_llm
from app.tools import (
    RETRIEVAL_DESCRIPTION,
    TOOL_OPTIONS,
    TOOLS,
    AvailableTools,
    get_retrieval_tool,
)


class AgentType(str, Enum):
    GPT_35_TURBO = "GPT 3.5 Turbo"
    GPT_4 = "GPT 4"
    AZURE_OPENAI = "GPT 4 (Azure OpenAI)"
    CLAUDE2 = "Claude 2"
    BEDROCK_CLAUDE2 = "Claude 2 (Amazon Bedrock)"
    GEMINI = "GEMINI"


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."


def get_agent_executor(
    tools: list,
    agent: AgentType,
    system_message: str,
):
    checkpointer = RedisCheckpoint()
    if agent == AgentType.GPT_35_TURBO:
        llm = get_openai_llm()
        return get_openai_agent_executor(tools, llm, system_message, checkpointer)
    elif agent == AgentType.GPT_4:
        llm = get_openai_llm(gpt_4=True)
        return get_openai_agent_executor(tools, llm, system_message, checkpointer)
    elif agent == AgentType.AZURE_OPENAI:
        llm = get_openai_llm(azure=True)
        return get_openai_agent_executor(tools, llm, system_message, checkpointer)
    elif agent == AgentType.CLAUDE2:
        llm = get_anthropic_llm()
        return get_xml_agent_executor(tools, llm, system_message, checkpointer)
    elif agent == AgentType.BEDROCK_CLAUDE2:
        llm = get_anthropic_llm(bedrock=True)
        return get_xml_agent_executor(tools, llm, system_message, checkpointer)
    elif agent == AgentType.GEMINI:
        llm = get_google_llm()
        return get_google_agent_executor(tools, llm, system_message, checkpointer)
    else:
        raise ValueError("Unexpected agent type")


class ConfigurableAgent(RunnableBinding):
    tools: Sequence[str]
    agent: AgentType
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    retrieval_description: str = RETRIEVAL_DESCRIPTION
    assistant_id: Optional[str] = None
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        tools: Sequence[str],
        agent: AgentType = AgentType.GPT_35_TURBO,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        retrieval_description: str = RETRIEVAL_DESCRIPTION,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)
        _tools = []
        for _tool in tools:
            if _tool == AvailableTools.RETRIEVAL:
                if assistant_id is None:
                    raise ValueError(
                        "assistant_id must be provided if Retrieval tool is used"
                    )
                _tools.append(get_retrieval_tool(assistant_id, retrieval_description))
            else:
                _tools.append(TOOLS[_tool]())
        _agent = get_agent_executor(_tools, agent, system_message)
        agent_executor = _agent.with_config({"recursion_limit": 50})
        super().__init__(
            tools=tools,
            agent=agent,
            system_message=system_message,
            retrieval_description=retrieval_description,
            bound=agent_executor,
            kwargs=kwargs or {},
            config=config or {},
        )


agent = (
    ConfigurableAgent(
        agent=AgentType.GPT_35_TURBO,
        tools=[],
        system_message=DEFAULT_SYSTEM_MESSAGE,
        retrieval_description=RETRIEVAL_DESCRIPTION,
        assistant_id=None,
    )
    .configurable_fields(
        agent=ConfigurableField(id="agent_type", name="Agent Type"),
        system_message=ConfigurableField(id="system_message", name="System Message"),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        tools=ConfigurableFieldMultiOption(
            id="tools",
            name="Tools",
            options=TOOL_OPTIONS,
            default=[],
        ),
        retrieval_description=ConfigurableField(
            id="retrieval_description", name="Retrieval Description"
        ),
    )
    .with_types(input_type=Sequence[AnyMessage], output_type=Sequence[AnyMessage])
)

if __name__ == "__main__":
    import asyncio

    from langchain.schema.messages import HumanMessage

    async def run():
        async for m in agent.astream_events(
            HumanMessage(content="whats your name"),
            config={"configurable": {"user_id": "2", "thread_id": "test1"}},
            version="v1",
        ):
            print(m)

    asyncio.run(run())