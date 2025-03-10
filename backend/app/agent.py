import pickle
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
)
from langgraph.checkpoint import CheckpointAt
from langgraph.graph.message import Messages
from langgraph.pregel import Pregel

from app.agent_types.tools_agent import get_tools_agent_executor
from app.chatbot import get_chatbot_executor
from app.checkpoint import PostgresCheckpoint
from app.llms import get_ollama_llm
from app.retrieval import get_retrieval_executor
from app.tools import (
    RETRIEVAL_DESCRIPTION,
    TOOLS,
    AvailableTools,
    Retrieval,
    Wikipedia,
    get_retrieval_tool,
    get_retriever,
)

Tool = Union[
    Wikipedia,
    Retrieval,
]

class LLMType(str, Enum):
    OLLAMA = "Ollama"

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

CHECKPOINTER = PostgresCheckpoint(serde=pickle, at=CheckpointAt.END_OF_STEP)

class ConfigurableSystem(RunnableBinding):
    mode: str
    llm_type: LLMType = LLMType.OLLAMA
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    tools: Optional[Sequence[Tool]] = None
    interrupt_before_action: bool = False
    retrieval_description: str = RETRIEVAL_DESCRIPTION
    user_id: Optional[str] = None

    def __init__(
        self,
        *,
        mode: str = "agent",
        llm_type: LLMType = LLMType.OLLAMA,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
        interrupt_before_action: bool = False,
        retrieval_description: str = RETRIEVAL_DESCRIPTION,
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        **others: Any,
    ) -> None:
        others.pop("bound", None)

        llm = get_ollama_llm() 

        if mode == "chatbot":
            executor = get_chatbot_executor(llm, system_message, CHECKPOINTER)
            
        elif mode == "retrieval":
            retriever = get_retriever(assistant_id, thread_id)
            executor = get_retrieval_executor(llm, retriever, system_message, CHECKPOINTER)

        elif mode == "agent":
            _tools = []
            if tools:
                for _tool in tools:
                    if _tool["type"] == AvailableTools.RETRIEVAL:
                        if assistant_id is None or thread_id is None:
                            raise ValueError(
                                "Both assistant_id and thread_id must be provided if Retrieval tool is used"
                            )
                    _tools.append(
                        get_retrieval_tool(assistant_id, thread_id, retrieval_description)
                    )
                else:
                    tool_config = _tool.get("config", {})
                    _returned_tools = TOOLS[_tool["type"]](**tool_config)
                    if isinstance(_returned_tools, list):
                        _tools.extend(_returned_tools)
                    else:
                        _tools.append(_returned_tools)

            agent_executor = get_tools_agent_executor(
                _tools, llm, system_message, interrupt_before_action, CHECKPOINTER
            )
            executor = agent_executor.with_config({"recursion_limit": 50})
        else:
            raise ValueError("Invalid mode. Must be one of: 'chatbot', 'retrieval', 'agent'")

        super().__init__(
            mode=mode,
            llm_type=llm_type,
            system_message=system_message,
            assistant_id=assistant_id,
            thread_id=thread_id,
            tools=tools,
            interrupt_before_action=interrupt_before_action,
            retrieval_description=retrieval_description,
            bound=executor,
            kwargs=kwargs or {},
            config=config or {},
        )


chatbot = (
    ConfigurableSystem(mode="chatbot", llm_type=LLMType.OLLAMA, checkpoint=CHECKPOINTER)
    .configurable_fields(
        # llm_type=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
)


chat_retrieval = (
    ConfigurableSystem(mode="retrieval", llm_type=LLMType.OLLAMA, checkpoint=CHECKPOINTER)
    .configurable_fields(
        # llm_type=ConfigurableField(id="llm_type", name="LLM Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(id="thread_id", name="Thread ID", is_shared=True),
    )
    .with_types(
        input_type=Dict[str, Any],
        output_type=Dict[str, Any],
    )
)


agent: Pregel = (
    ConfigurableSystem(
        mode="agent",
        llm_type=LLMType.OLLAMA,
        tools=[],
        system_message=DEFAULT_SYSTEM_MESSAGE,
        retrieval_description=RETRIEVAL_DESCRIPTION,
        assistant_id=None,
        thread_id=None,
    )
    .configurable_fields(
        # llm_type=ConfigurableField(id="agent_type", name="Agent Type"),
        system_message=ConfigurableField(id="system_message", name="Instructions"),
        # interrupt_before_action=ConfigurableField(
        #     id="interrupt_before_action",
        #     name="Tool Confirmation",
        #     description="If Yes, you'll be prompted to continue before each tool is executed.\nIf No, tools will be executed automatically by the agent.",
        # ),
        assistant_id=ConfigurableField(
            id="assistant_id", name="Assistant ID", is_shared=True
        ),
        thread_id=ConfigurableField(id="thread_id", name="Thread ID", is_shared=True),
        tools=ConfigurableField(id="tools", name="Tools"),
        retrieval_description=ConfigurableField(
            id="retrieval_description", name="Retrieval Description"
        ),
    )
    .configurable_alternatives(
        ConfigurableField(id="type", name="Bot Type"),
        default_key="agent",
        prefix_keys=True,
        chatbot=chatbot,
        chat_retrieval=chat_retrieval,
    )
    .with_types(
        input_type=Messages,
        output_type=Sequence[AnyMessage],
    )
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
