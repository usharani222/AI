import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found")

model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    streaming=True
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = model.invoke(messages)

print("Response:")
print(response.content)

print("\nUsage:")
print(response.usage_metadata)

print("\nMetadata:")
print(response.response_metadata)