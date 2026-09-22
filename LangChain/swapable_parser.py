import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)

load_dotenv()


# ============================================================
# MODEL
# ============================================================

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    Explain {topic}.

    Include:
    - what it is
    - its main use
    - one important feature
    """
)


# ============================================================
# PARSER 1 — STRING
# ============================================================

string_parser = StrOutputParser()

chain = prompt | model | string_parser

result = chain.invoke({
    "topic": "Redis"
})

print("=" * 70)
print("STRING PARSER")
print("=" * 70)

print(type(result))
print(result)


# ============================================================
# PARSER 2 — PYDANTIC
# ============================================================

class Technology(BaseModel):
    name: str = Field(description="Name of the technology")
    description: str = Field(description="What the technology is")
    main_use: str = Field(description="Main use")
    important_feature: str = Field(description="One important feature")


pydantic_parser = PydanticOutputParser(
    pydantic_object=Technology
)


structured_prompt = ChatPromptTemplate.from_template(
    """
    Explain {topic}.

    Return the result according to these instructions:

    {format_instructions}
    """
).partial(
    format_instructions=pydantic_parser.get_format_instructions()
)

chain = structured_prompt | model | pydantic_parser

result = chain.invoke({
    "topic": "Redis"
})

print("\n" + "=" * 70)
print("PYDANTIC PARSER")
print("=" * 70)

print(type(result))
print(result)

print("\nName:", result.name)
print("Description:", result.description)
print("Main use:", result.main_use)
print("Important feature:", result.important_feature)