from dotenv import load_dotenv
from openai import OpenAI
import os
from pydantic import BaseModel

load_dotenv()

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class Response(BaseModel):
    question: str
    answer: str
    usage: Usage

api_key=os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("API Key missing")

client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
prompt=input("Enter prompt:")
response = (client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role":"user",
            "content":prompt,
        }
    ],
    extra_body={
        "usage":{
            "include":True
        }
    }
))


# print("Raw response:")
# print(response)
# print("="*250+"\n")

# print("Usage")
# if response.usage:
#     print("Completion usage",response.usage.completion_tokens)
#     print("Prompt usage",response.usage.prompt_tokens)
#     print("Total usage",response.usage.total_tokens)


# print("\n" + "=" * 70)
# print("MODEL RESPONSE")
# print("=" * 70)

# print(response.choices[0].message.content)


result = Response(
    question=prompt,
    answer=response.choices[0].message.content,
    usage=Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        cost=response.usage.cost
    )
)


print(result)