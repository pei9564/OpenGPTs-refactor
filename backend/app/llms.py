import os
from functools import lru_cache
import structlog
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Type,
    Union,
)
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput

logger = structlog.get_logger(__name__)

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""
class CustomChatOllama(ChatOllama):
    """Function chat model that uses Ollama API."""

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    
    def convert_to_ollama_tool(self, tool: Any) -> Dict:
        """Convert a tool to an Ollama tool."""
        def __is_pydantic_class(obj: Any) -> bool:
            return isinstance(obj, type) and (
                issubclass(obj, BaseModel) or BaseModel in obj.__bases__
            )
    
        if __is_pydantic_class(tool):
            schema = tool.construct().schema()
            definition = {"name": schema["title"], "properties": schema["properties"]}
            if "required" in schema:
                definition["required"] = schema["required"]
            return definition
        
        elif isinstance(tool, BaseTool):
            schema = tool.args_schema.schema() if tool.args_schema else {"properties": {}}
            return {
                "name": tool.name,
                "description": tool.description,
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        
        raise ValueError(
            f"Cannot convert {tool} to an Ollama tool. {tool} needs to be a Pydantic model."
        )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        ollama_tools = [self.convert_to_ollama_tool(tool) for tool in tools]
        return self.bind(functions=ollama_tools, **kwargs)

@lru_cache(maxsize=1)
def get_ollama_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return CustomChatOllama(model=model_name, base_url=ollama_base_url)
