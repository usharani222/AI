import os
import json
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
)

# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")


# ============================================================
# 2. MODEL
# ============================================================

model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 3. STR OUTPUT PARSER
# ============================================================

print("\n" + "=" * 70)
print("1. StrOutputParser")
print("=" * 70)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "topic": "Redis"
})

print("Type:", type(result))
print("Result:")
print(result)


# ============================================================
# 4. COMMA SEPARATED LIST OUTPUT PARSER
# ============================================================

print("\n" + "=" * 70)
print("2. CommaSeparatedListOutputParser")
print("=" * 70)

list_parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    Give me 5 programming languages suitable for backend development.

    Return ONLY a comma-separated list.
    """
)

chain = prompt | model | list_parser

result = chain.invoke({})

print("Type:", type(result))
print("Result:")
print(result)

for language in result:
    print("-", language)


# ============================================================
# 5. JSON OUTPUT PARSER
# ============================================================

print("\n" + "=" * 70)
print("3. JsonOutputParser")
print("=" * 70)

json_parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    Return information about Redis.

    Return ONLY valid JSON with these fields:
    name
    type
    use_case
    in_memory

    {format_instructions}
    """
).partial(
    format_instructions=json_parser.get_format_instructions()
)

chain = prompt | model | json_parser

result = chain.invoke({})

print("Type:", type(result))
print("Result:")
print(result)

print("\nName:", result["name"])
print("Type:", result["type"])
print("Use case:", result["use_case"])
print("In memory:", result["in_memory"])


# ============================================================
# 6. PYDANTIC OUTPUT PARSER
# ============================================================

print("\n" + "=" * 70)
print("4. PydanticOutputParser")
print("=" * 70)


class Technology(BaseModel):
    name: str = Field(description="Name of the technology")
    category: str = Field(description="Technology category")
    primary_use: str = Field(description="Main use of the technology")
    difficulty: str = Field(description="Beginner, Intermediate, or Advanced")


pydantic_parser = PydanticOutputParser(
    pydantic_object=Technology
)

prompt = ChatPromptTemplate.from_template(
    """
    Give information about the technology Redis.

    {format_instructions}
    """
).partial(
    format_instructions=pydantic_parser.get_format_instructions()
)

chain = prompt | model | pydantic_parser

result = chain.invoke({})

print("Type:", type(result))
print("Result:")
print(result)

print("\nName:", result.name)
print("Category:", result.category)
print("Primary use:", result.primary_use)
print("Difficulty:", result.difficulty)


# ============================================================
# 7. XML OUTPUT PARSER
# ============================================================

print("\n" + "=" * 70)
print("5. XMLOutputParser")
print("=" * 70)

xml_parser = XMLOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    Give information about Python.

    Return XML with:
    - name
    - category
    - primary_use

    {format_instructions}
    """
).partial(
    format_instructions=xml_parser.get_format_instructions()
)

chain = prompt | model | xml_parser

result = chain.invoke({})

print("Type:", type(result))
print("Result:")
print(result)


# ============================================================
# 8. RAW AI MESSAGE vs PARSED OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("6. Raw AIMessage vs Parsed Output")
print("=" * 70)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one paragraph."
)

# Without parser
raw_chain = prompt | model

raw_result = raw_chain.invoke({
    "topic": "Docker"
})

print("\nWITHOUT parser:")
print("Type:", type(raw_result))
print("Content:", raw_result.content)


# With StrOutputParser
parsed_chain = prompt | model | StrOutputParser()

parsed_result = parsed_chain.invoke({
    "topic": "Docker"
})

print("\nWITH StrOutputParser:")
print("Type:", type(parsed_result))
print("Content:", parsed_result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PARSER SUMMARY")
print("=" * 70)

print("""
1. StrOutputParser
   AIMessage → string

2. CommaSeparatedListOutputParser
   LLM text → Python list

3. JsonOutputParser
   LLM JSON → Python dictionary

4. PydanticOutputParser
   LLM structured data → Pydantic object

5. XMLOutputParser
   LLM XML → Python structured representation

General pattern:

Prompt → Model → Parser → Application

Example:

prompt | model | parser
""")