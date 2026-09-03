import os
from collections.abc import Iterator
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

CLOUD_RU_API_KEY_ENV = "CLOUD_RU_API_KEY"

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather in a city."""
    return f"In {city} it is sunny, +22 C"


@lru_cache(maxsize=1)
def get_tool_llm():
    api_key = os.getenv(CLOUD_RU_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {CLOUD_RU_API_KEY_ENV} before starting the app")

    return ChatOpenAI(
        base_url="https://foundation-models.api.cloud.ru/v1",
        model="openai/gpt-oss-120b",
        api_key=api_key,
        streaming=True,
    ).bind_tools([get_weather])


@lru_cache(maxsize=1)
def get_streaming_llm():
    api_key = os.getenv(CLOUD_RU_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {CLOUD_RU_API_KEY_ENV} before starting the app")

    return ChatOpenAI(
        base_url="https://foundation-models.api.cloud.ru/v1",
        model="openai/gpt-oss-120b",
        api_key=api_key,
        streaming=True,
    )


def call_model(state: MessagesState, config: RunnableConfig):
    llm = get_tool_llm()
    return {"messages": [llm.invoke(state["messages"], config=config)]}


def call_stream_model(state: MessagesState, config: RunnableConfig):
    llm = get_streaming_llm()
    return {"messages": [llm.invoke(state["messages"], config=config)]}


graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

graph_app = graph.compile()

stream_graph = StateGraph(MessagesState)
stream_graph.add_node("agent", call_stream_model)
stream_graph.add_edge(START, "agent")
stream_graph_app = stream_graph.compile()

app = FastAPI()


def chunk_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    return ""


def stream_graph_response(prompt: str) -> Iterator[str]:
    try:
        for msg_chunk, _metadata in graph_app.stream(
            {"messages": [("user", prompt)]},
            stream_mode="messages",
        ):
            text = chunk_content_to_text(msg_chunk.content)
            if text:
                yield text
    except Exception as exc:
        yield f"\n[stream error: {type(exc).__name__}: {exc}]\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stream")
async def stream(prompt: str) -> StreamingResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    return StreamingResponse(
        stream_graph_response(prompt),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
