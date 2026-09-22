from langchain_core.messages.block_translators import anthropic
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================
# MODEL
# ============================================================

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()


# ============================================================
# 1. RUNNABLE SEQUENCE
# ============================================================

prompt = PromptTemplate.from_template(
    "Explain about {topic} in one line."
)

chain = prompt | model | parser

result = chain.invoke({
    "topic": "AI"
})

print("Runnable Sequence:\n")
print(result)
print("=" * 100)


# ============================================================
# 2. RUNNABLE PASSTHROUGH
# ============================================================

prompt = PromptTemplate.from_template(
    "Explain about {topic} in one line."
)

passthrough_chain = prompt | model | parser

chain = RunnableParallel(
    topic=RunnablePassthrough(),
    summary=passthrough_chain
)

result = chain.invoke({
    "topic": "Docker"
})

print("Runnable Passthrough:\n")
print(result)
print("=" * 100)


# ============================================================
# 3. RUNNABLE PARALLEL
# ============================================================

prompt1 = PromptTemplate.from_template(
    "Explain {topic} in one line."
)

prompt2 = PromptTemplate.from_template(
    "Explain {topic} to a first-class student in one line."
)

prompt3 = PromptTemplate.from_template(
    "Explain {topic} to a software engineer in one line."
)

parallel_chain = RunnableParallel(
    normal=prompt1 | model | parser,
    student=prompt2 | model | parser,
    engineer=prompt3 | model | parser
)

result = parallel_chain.invoke({
    "topic": "Generative AI"
})

print("Runnable Parallel:\n")

print("Normal:")
print(result["normal"])

print("\nFirst-class student:")
print(result["student"])

print("\nEngineer:")
print(result["engineer"])

print("=" * 100)