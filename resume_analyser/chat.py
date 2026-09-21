import os
import time
import json
import re
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

# Load environment variables from .env file
load_dotenv()

# Schema with defensive defaults to prevent crashes on messy resumes
class ChatResponse(BaseModel):
    name: str = "Unknown"
    email: Optional[str] = None
    skills: list[str] = []
    years_experience: int = 0

# Retrieve OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("API Key missing")

# OpenRouter client configured with a 15-second timeout
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=15.0
)

# Calls LLM using exponential backoff retry for network/API failures
def call_llm_with_retry(prompt: str, max_attempts: int = 3) -> str:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[LLM] Attempt {attempt}/{max_attempts}")
            response = client.chat.completions.create(
                model="inclusionai/ling-3.0-flash-vl:free",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Extract information from the following resume.

                        Return ONLY valid JSON with exactly these fields:
                        {{
                            "name": string,
                            "email": string,
                            "skills": list of strings,
                            "years_experience": integer
                        }}

                        Resume:
                        {prompt}
                        """
                    }
                ]
            )
            content = response.choices[0].message.content
            # Ensure output contains a valid JSON block before accepting
            if content and re.search(r"\{[\s\S]*\}", content):
                return content
            
            # Retry if the model produced text without JSON
            print(f"[LLM] Non-JSON response on attempt {attempt}: {repr(content)[:100]}")
            if attempt == max_attempts:
                raise ValueError("Model failed to output a valid JSON block.")
            time.sleep(1)

        except APITimeoutError as e:
            # Retry on timeout using exponential backoff
            print(f"[LLM] Request timed out on attempt {attempt}")
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(f"Request timed out after {max_attempts} attempts.") from e
            wait_time = 2 ** (attempt - 1)
            time.sleep(wait_time)

        except APIConnectionError as e:
            # Retry on network drop using exponential backoff
            print(f"[LLM] Connection error on attempt {attempt}")
            last_error = e
            if attempt == max_attempts:
                raise RuntimeError(f"Connection failed after {max_attempts} attempts.") from e
            wait_time = 2 ** (attempt - 1)
            time.sleep(wait_time)

        except APIStatusError as e:
            status_code = e.status_code
            print(f"[LLM] API returned HTTP {status_code} on attempt {attempt}")
            last_error = e
            # Retry only on 429 rate limit or 5xx server errors
            if status_code == 429 or status_code in (500, 502, 503, 504):
                if attempt == max_attempts:
                    raise RuntimeError(f"API error ({status_code}) after {max_attempts} attempts.") from e
                wait_time = 2 ** (attempt - 1)
                time.sleep(wait_time)
            else:
                # Fail immediately on 4xx client errors that cannot be retried
                raise RuntimeError(f"Non-retryable API error: HTTP {status_code}") from e

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_error}")

# Extracts resume data and validates it against Pydantic schema
def chat(prompt: str) -> ChatResponse:
    raw_content = call_llm_with_retry(prompt)

    # Isolate JSON object even if enclosed in markdown fences
    match = re.search(r"\{[\s\S]*\}", raw_content)
    if not match:
        raise ValueError("Could not extract a valid JSON structure from the response.")

    json_str = match.group(0)
    data = json.loads(json_str)
    # Validate parsed dictionary into ChatResponse
    return ChatResponse.model_validate(data)

# Generator that streams response tokens in real-time
def stream_chat(prompt: str):
    response = client.chat.completions.create(
        model="inclusionai/ling-3.0-flash-vl:free",
        messages=[
            {
                "role": "user",
                "content": f"""
                Extract and summarize structured resume information in JSON format:
                {prompt}
                """
            }
        ],
        stream=True
    )
    # Yield tokens one-by-one as they arrive
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content
