import os
import time

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1/",
    api_key=api_key,
    timeout=5
)


def call_llm(prompt: str, max_attempts: int = 3):

    for attempt in range(1, max_attempts + 1):

        try:
            print(f"\nAttempt {attempt}/{max_attempts}")

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content

        # ------------------------------------------------
        # 1. TIMEOUT
        # ------------------------------------------------
        except APITimeoutError as e:

            print("Request timed out.")

            if attempt == max_attempts:
                raise RuntimeError(
                    f"Request timed out after {max_attempts} attempts."
                ) from e

            wait_time = 2 ** (attempt - 1)

            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


        # ------------------------------------------------
        # 2. CONNECTION ERROR
        # ------------------------------------------------
        except APIConnectionError as e:

            print("Could not connect to the API.")

            if attempt == max_attempts:
                raise RuntimeError(
                    f"Connection failed after {max_attempts} attempts."
                ) from e

            wait_time = 2 ** (attempt - 1)

            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


        # ------------------------------------------------
        # 3. HTTP STATUS ERRORS
        # ------------------------------------------------
        except APIStatusError as e:

            status_code = e.status_code

            print(f"API returned HTTP {status_code}")

            # --------------------------------------------
            # 429 → RATE LIMIT
            # --------------------------------------------
            if status_code == 429:

                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Rate limited after {max_attempts} attempts."
                    ) from e

                wait_time = 2 ** (attempt - 1)

                print(
                    f"Rate limited (429). "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)


            # --------------------------------------------
            # 5xx → SERVER ERROR
            # --------------------------------------------
            elif status_code in (500, 502, 503, 504):

                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Server error ({status_code}) "
                        f"after {max_attempts} attempts."
                    ) from e

                wait_time = 2 ** (attempt - 1)

                print(
                    f"Temporary server error ({status_code}). "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)


            # --------------------------------------------
            # OTHER HTTP ERRORS → DO NOT RETRY
            # --------------------------------------------
            else:

                raise RuntimeError(
                    f"Non-retryable API error: HTTP {status_code}"
                ) from e


# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------

prompt = input("Enter prompt: ")

try:

    answer = call_llm(prompt)

    print("\nResponse:")
    print(answer)

except RuntimeError as e:

    print(f"\nFinal error: {e}")