"""LangSmith tracing and observability. Traces LangGraph, Bedrock, Pinecone, and ingestion."""
import os
import logging
from functools import wraps

log = logging.getLogger(__name__)

def initialize():
    """Call once, before instrumented app modules are loaded."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() in ("1", "true", "yes") or \
                      os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("1", "true", "yes")

    if not api_key or not tracing_enabled:
        log.info("LangSmith observability disabled (no key or tracing disabled)")
        return

    # Set standard LangChain / LangSmith environment variables
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "documind")
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    log.info("LangSmith observability enabled for project '%s'", project)

def trace(name):
    """Decorate sync/async functions; traces runs in LangSmith."""
    def decorate(fn):
        try:
            from langsmith import traceable
            return traceable(name=name)(fn)
        except Exception:
            return fn
    return decorate
