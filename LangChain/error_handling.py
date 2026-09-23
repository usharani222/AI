import os
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool

load_dotenv()


# --------------------------------------------------
# 1. Tool (with forced exception)
# --------------------------------------------------

@tool
def get_service_status(service_name: str) -> str:
    """Check the real-time operational status of an internal service."""
    service = service_name.lower().strip()

    # Simulate a tool failure when checking payment or database
    if "payment" in service or "database" in service:
        raise ConnectionResetError(f"Connection timeout while connecting to {service_name}.")

    statuses = {
        "redis": "Operational (0% packet loss, healthy)",
        "postgresql": "Operational (Active connections: 28/100)",
    }
    return statuses.get(service, f"Service '{service_name}' is healthy.")


tools = [get_service_status]


# --------------------------------------------------
# 2. Model & Retry
# --------------------------------------------------

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Bind tools and add retry (retries once on failure)
model_with_tools = model.bind_tools(tools).with_retry(stop_after_attempt=2)


# --------------------------------------------------
# 3. Tool Execution with Retry & Fallback
# --------------------------------------------------

def execute_tool_with_fallback(message):
    if not (hasattr(message, "tool_calls") and message.tool_calls):
        return getattr(message, "content", str(message))

    call = message.tool_calls[0]
    tool_name = call["name"]
    args = call["args"]

    print(f"Tool called: {tool_name}")
    print(f"Arguments: {args}")

    # Attempt 1
    try:
        if tool_name == "get_service_status":
            return get_service_status.invoke(args)
    except Exception as e:
        print(f"Attempt 1 failed: {e}")
        print("Retrying once...")
        time.sleep(0.5)

        # Attempt 2 (Retry)
        try:
            return get_service_status.invoke(args)
        except Exception as retry_error:
            print(f"Attempt 2 failed: {retry_error}")
            print("Activating canned fallback response.")
            # Canned safe response
            return "The requested service is temporarily unavailable. Please try again later."


# --------------------------------------------------
# 4. Chain with Fallback
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. If asked about service status, call get_service_status."),
    ("human", "{question}"),
])

primary_chain = prompt | model_with_tools | RunnableLambda(execute_tool_with_fallback)

# Canned fallback if the entire chain fails
system_fallback = RunnableLambda(
    lambda x: "Our service is currently experiencing technical difficulties. Please check back later."
)

chain = primary_chain.with_fallbacks([system_fallback])


# --------------------------------------------------
# 5. Tests
# --------------------------------------------------

print("\n===== TEST 1: Normal Tool Call =====")
res1 = chain.invoke({"question": "What is the status of redis?"})
print("Result:", res1)

print("\n===== TEST 2: Forced Tool Failure (Degrades Gracefully) =====")
res2 = chain.invoke({"question": "What is the status of payment service?"})
print("Result:", res2)
