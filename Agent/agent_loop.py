# One LLM + one calculator tool + one loop

## 1. LLM
from langchain_core.messages import ToolMessage
from openai.types.responses import response
from os import getenv
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(
    model="openrouter/free",
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

## 2. Tool
from langchain_core.tools import tool
@tool
def calculator(exp:str)->str:
    """For calculation of mathmatical expressions."""
    try:
        return str(eval(exp))
    except Exception as e:
        return f"Error: {e}"
llm_tools=model.bind_tools([calculator])

## 3. Loop
from langchain_core.prompts import message
message = [
    ("user","calculate 125*2/2")
]
while True:
    response=llm_tools.invoke(message)
    message.append(response)
    print(f"Message appended{response.content}")
    if not response.tool_calls:
        print(response.content)
        break
    for tool_call in response.tool_calls:
        if tool_call["name"] == "calculator":
            result = calculator.invoke(tool_call["args"])
            message.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )


