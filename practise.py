import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


load_dotenv()


# --------------------------------------------------
# 1. Define schema
# --------------------------------------------------

class Resume(BaseModel):
    name: str
    email: str
    skills: list[str]
    years_experience: int


# --------------------------------------------------
# 2. Get API key
# --------------------------------------------------

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing")


# --------------------------------------------------
# 3. Create OpenRouter client
# --------------------------------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# --------------------------------------------------
# 4. Resume input
# --------------------------------------------------

resume_text = """
John Doe is a Software Engineer with 2 years of experience.
He has worked with Python, FastAPI, PostgreSQL, Docker and Redis.

Email: john.doe@gmail.com
"""


# --------------------------------------------------
# 5. Ask LLM to extract structured information
# --------------------------------------------------

prompt = f"""
Extract information from the following resume.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "name": string,
    "email": string,
    "skills": list of strings,
    "years_experience": integer
}}

Resume:

{resume_text}
"""


response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 6. Get model's response
# --------------------------------------------------

raw_output = response.choices[0].message.content

print("=" * 70)
print("RAW LLM OUTPUT")
print("=" * 70)

print(raw_output)


# --------------------------------------------------
# 7. Validate using Pydantic
# --------------------------------------------------

print("\n" + "=" * 70)
print("PYDANTIC VALIDATION")
print("=" * 70)

try:
    resume = Resume.model_validate_json(raw_output)

    print("Validation successful!")
    print(resume)

except ValidationError as e:
    print("Validation failed!")
    print(e)