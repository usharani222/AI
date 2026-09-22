import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()


model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


@tool
def calculator(a: int, b: int) -> int:
    """Add two integers together and return their sum."""
    return a + b


@tool
def search_events(query: str) -> list[str]:
    """Search the event catalog for events matching the user's search query."""
    
    events = {
        "concert": [
            "Sunburn Music Festival - Vizag",
            "Live Indie Night - Hyderabad",
        ],
        "tech": [
            "AI & Machine Learning Summit - Bangalore",
            "Developer Conference - Hyderabad",
        ],
        "sports": [
            "Vizag Marathon 2026",
            "Cricket Championship - Hyderabad",
        ],
    }

    query = query.lower()

    for keyword, results in events.items():
        if keyword in query:
            return results

    return ["No matching events found."]


tools = [calculator, search_events]

model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke(
    "What concerts are available?"
)

tool_calls = response.tool_calls

tool = tool_calls[0]["name"]
args = tool_calls[0]["args"]

print(tool)
print(args)