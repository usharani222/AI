from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key=os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("API key missing")

client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
prompt=input("Enter your prompt:")
response=client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role":"user",
            "content":prompt
        }
    ],
    stream=True
)

print("\nResponse:\n")

for chunk in response:
    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)

print()