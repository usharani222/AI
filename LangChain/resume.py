
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


load_dotenv()


# -------------------------
# Model
# -------------------------

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# -------------------------
# Resume Schema
# -------------------------

class Resume(BaseModel):
    name: str = Field(
        description="Candidate's full name"
    )

    email: str = Field(
        description="Candidate's email address"
    )

    skills: list[str] = Field(
        description="Programming languages, frameworks and technologies"
    )

    education: str = Field(
        description="Highest education qualification"
    )

    experience: str = Field(
        description="Brief description of work experience"
    )


# -------------------------
# Structured Output
# -------------------------

structured_model = model.with_structured_output(Resume)


# -------------------------
# Resume Text
# -------------------------

resume_text = """
Usha Rani
Email: usha@example.com

B.Tech in Computer Science and Engineering.

Skills:
Python, Java, C++, FastAPI, React, PostgreSQL, Redis, Docker.

Experience:
Software development internship working with FastAPI,
Redis and authentication systems.
"""


# -------------------------
# Invoke
# -------------------------

result = structured_model.invoke(
    f"""
    Extract the candidate information from the following resume.

    Resume:
    {resume_text}
    """
)


# -------------------------
# Result
# -------------------------

print("TYPE:")
print(type(result))

print("\nRESULT:")
print(result)

print("\nNAME:")
print(result.name)

print("\nEMAIL:")
print(result.email)

print("\nSKILLS:")
print(result.skills)

print("\nEDUCATION:")
print(result.education)

print("\nEXPERIENCE:")
print(result.experience)