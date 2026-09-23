import asyncio
import os
import sys

from dotenv import load_dotenv

# Ensure Windows terminal handles Unicode characters/emojis without crashing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool


load_dotenv()


# ============================================================
# 1. MODEL
# ============================================================

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 2. BASIC CHAIN
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI teacher. Explain concepts simply and briefly."
    ),
    (
        "human",
        "{question}"
    ),
])

chain = prompt | llm


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def get_service_status(service_name: str) -> str:
    """Check the real-time operational status and health of an internal service (e.g., redis, postgresql, auth)."""
    service = service_name.lower().strip()
    status_db = {
        "redis": "Operational (0% packet loss, 12ms latency, memory 42% used)",
        "postgresql": "Operational (Active connections: 34/100, healthy)",
        "auth": "Operational (Active sessions: 1,420, 0 auth failures)",
    }
    return status_db.get(service, f"Service '{service_name}' status is currently unknown or offline.")


tools = [get_service_status]
llm_with_tools = llm.bind_tools(tools)


def execute_tools(message):
    """Executes any tool calls requested by the model during the chain."""
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            if t_name == "get_service_status":
                return get_service_status.invoke(t_args)
    return getattr(message, "content", str(message))


tool_chain = prompt | llm_with_tools | RunnableLambda(execute_tools)


# ============================================================
# DEMO 1: stream()
# ============================================================

def demo_stream():
    print("\n" + "=" * 70)
    print("1. stream() — Synchronous Token Streaming")
    print("=" * 70)

    print("\nQuestion: What is LangChain?\n")
    print("Response: ", end="", flush=True)

    # stream() returns chunks synchronously
    for chunk in chain.stream({
        "question": "What is LangChain? Explain in 3 sentences."
    }):
        print(chunk.content, end="", flush=True)

    print("\n")


# ============================================================
# DEMO 2: astream()
# ============================================================

async def demo_astream():
    print("\n" + "=" * 70)
    print("2. astream() — Asynchronous Token Streaming")
    print("=" * 70)

    print("\nQuestion: What is RAG?\n")
    print("Response: ", end="", flush=True)

    # astream() returns chunks asynchronously
    async for chunk in chain.astream({
        "question": "What is RAG? Explain in 3 sentences."
    }):
        print(chunk.content, end="", flush=True)

    print("\n")


# ============================================================
# DEMO 3: astream_events()
# ============================================================

async def demo_astream_events():
    print("\n" + "=" * 70)
    print("3. astream_events() — Raw Event Inspection")
    print("=" * 70)

    print("\nQuestion: What is an AI agent?\n")
    print("Events:\n")

    async for event in chain.astream_events(
        {
            "question": "What is an AI agent? Explain briefly."
        },
        version="v2",
    ):
        event_type = event["event"]
        event_name = event["name"]

        # Chain lifecycle
        if event_type == "on_chain_start":
            print(f"[CHAIN START] {event_name}")
        elif event_type == "on_chat_model_start":
            print(f"[LLM START] {event_name}")
        elif event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(f"[TOKEN] {chunk.content!r}")
        elif event_type == "on_chat_model_end":
            print(f"[LLM END] {event_name}")
        elif event_type == "on_chain_end":
            print(f"[CHAIN END] {event_name}")

    print()


# ============================================================
# DEMO 4: STREAMING WITH TOOLS  
# ============================================================

async def stream_chat_with_tools(question: str):
    """
    Streams output incrementally and visually distinguishes tool-call events.
    """
    print(f"\nUser Query: {question}")
    print("Assistant Response: ", end="", flush=True)

    async for event in tool_chain.astream_events(
        {"question": question},
        version="v2"
    ):
        event_type = event["event"]

        # 1. Incrementally stream user-facing tokens
        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)

        # 2. Visibly distinguish tool call start
        elif event_type == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input")
            print(f"\n\n[TOOL CALL: {tool_name}]")
            print(f"Arguments: {tool_input}", flush=True)

        # 3. Visibly distinguish tool call result
        elif event_type == "on_tool_end":
            tool_output = event["data"].get("output")
            print(f"[TOOL RESULT: {tool_output}]\n", flush=True)

    print("\n" + "-" * 70)


async def demo_streaming_with_tools():
    print("\n" + "=" * 70)
    print("4. Day 23 Mini Practice: Token & Tool Event Streaming")
    print("=" * 70)

    # Test A: Direct token streaming (no tool needed)
    await stream_chat_with_tools("Explain what Python is in two sentences.")

    # Test B: Tool call event streaming (triggers get_service_status)
    await stream_chat_with_tools("What is the real-time operational status of Redis?")


# ============================================================
# MAIN
# ============================================================

async def main():
    # 1. Synchronous streaming
    demo_stream()

    # 2. Asynchronous streaming
    await demo_astream()

    # 3. Event streaming inspection
    # await demo_astream_events()  # (uncomment if you want to inspect raw tokens)

    # 4. Day 23: Token Streaming + Tool Call Event Streaming
    await demo_streaming_with_tools()


if __name__ == "__main__":
    asyncio.run(main())
