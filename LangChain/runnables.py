from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful technical teacher."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])

model=ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

chain = prompt | model

response = chain.invoke({
    "topic": "Redis"
})

print(response.content)