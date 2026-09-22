import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# -----------------------------
# Prompt Variant 1: Beginner
# -----------------------------

beginner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a teacher explaining technical concepts to complete beginners."
    ),
    (
        "human",
        "Explain {topic} using very simple language and one real-world analogy."
    )
])


# -----------------------------
# Prompt Variant 2: Interview
# -----------------------------

interview_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior software engineer conducting a technical interview."
    ),
    (
        "human",
        "Explain {topic} as an interview answer. "
        "Include definition, why it is used, and one practical example."
    )
])


# -----------------------------
# Prompt Variant 3: Concise
# -----------------------------

concise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical documentation writer."
    ),
    (
        "human",
        "Explain {topic} in exactly 3 concise bullet points. "
        "Avoid unnecessary details."
    )
])


# -----------------------------
# Run all variants
# -----------------------------

prompts = [
    ("BEGINNER", beginner_prompt),
    ("INTERVIEW", interview_prompt),
    ("CONCISE", concise_prompt)
]

for name, prompt in prompts:

    messages = prompt.invoke({
        "topic": "Redis"
    })

    response = model.invoke(messages)

    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(response.content)