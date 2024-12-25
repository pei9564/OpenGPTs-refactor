import os
from functools import lru_cache
import structlog
from langchain_community.chat_models.ollama import ChatOllama

logger = structlog.get_logger(__name__)

@lru_cache(maxsize=1)
def get_ollama_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, base_url=ollama_base_url)
