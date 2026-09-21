# Resume Analyzer API — Architecture Documentation

FastAPI that extracts structured candidate profiles from unstructured resume text using Large Language Models (LLMs), enforced by Pydantic validation, exponential backoff retries, and strict timeout boundaries.

---

## 1. Quick Start & Setup Guide

Follow these exact steps to set up and run the service from scratch.

### Prerequisites
* Python 3.10+
* An OpenRouter API Key (get one free at [openrouter.ai](https://openrouter.ai/settings/keys))

### Installation
```bash
# 1. Clone or navigate to the project directory
cd resume_analyser

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install fastapi uvicorn openai pydantic python-dotenv
```

### Environment Configuration
Create a `.env` file in the `resume_analyser/` directory:
```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

### Running the API Server
```bash
python main.py
```
* **API Base URL**: `http://127.0.0.1:8000`
* **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
* **OpenAPI Schema**: `http://127.0.0.1:8000/openapi.json`

---

## 2. Request Flow Architecture

```
[ Client / Frontend ]
         │
         │  POST /analyze-resume (JSON Body: {"prompt": "..."})
         ▼
┌────────────────────────────────────────────────────────┐
│ FastAPI Gateway (main.py)                              │
│  ├─ Check 1: Is prompt empty or whitespace? ──► [400]  │
│  └─ Check 2: Is prompt < 15 characters?    ──► [400]  │
└────────────────────────┬───────────────────────────────┘
                         │ (Valid Input)
                         ▼
┌────────────────────────────────────────────────────────┐
│ Core Extraction Engine (chat.py)                       │
│                                                        │
│  ┌─ Retry Loop (Max 3 attempts, 15s timeout)           │
│  │   ├─ Exponential backoff: 1s -> 2s -> 4s            │
│  │   └─ Catches: 429, 5xx, Timeout, Connection drops   │
│  │                                                     │
│  │   Calls OpenRouter LLM Endpoint                     │
│  │   (inclusionai/ling-3.0-flash-vl:free)              │
│  └─────────────────────┬───────────────────────────────┘
                         │ (Raw LLM Output)
                         ▼
┌────────────────────────────────────────────────────────┐
│ Data Sanitization & Extraction                         │
│  ├─ Regex Isolation: Strips ```json ... ``` fences    │
│  └─ json.loads() conversion                            │
└────────────────────────┬───────────────────────────────┘
                         │ (Parsed Dict)
                         ▼
┌────────────────────────────────────────────────────────┐
│ Pydantic Validation (ChatResponse)                     │
│  ├─ Fields: name, email, skills, years_experience      │
│  └─ Defensive defaults for missing attributes          │
└────────────────────────┬───────────────────────────────┘
                         │ (Validated Pydantic Model)
                         ▼
             [ HTTP 200 OK Response ]
```

---

## 3. API Contract & Schema

### Endpoint 1: Analyze Resume (Synchronous)
* **Route**: `POST /analyze-resume`
* **Header**: `Content-Type: application/json`

#### Request Body Schema (`ResumeRequest`)
| Field | Type | Required | Description | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `prompt` | `string` | **Yes** | Raw resume text to analyze | Minimum 15 characters, non-empty |

```json
{
  "prompt": "John Doe is a Software Engineer with 2 years of experience. He works with Python, FastAPI, and Docker. Email: john.doe@gmail.com"
}
```

#### Success Response Schema (`ChatResponse` — HTTP 200 OK)
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | `"Unknown"` | Candidate full name |
| `email` | `string \| null` | `null` | Primary contact email address |
| `skills` | `list[string]` | `[]` | Extracted technical skills and tools |
| `years_experience` | `integer` | `0` | Estimated or stated years of professional experience |

```json
{
  "name": "John Doe",
  "email": "john.doe@gmail.com",
  "skills": ["Python", "FastAPI", "Docker"],
  "years_experience": 2
}
```

#### Error Responses
| HTTP Status | Reason | Example Response Body |
| :--- | :--- | :--- |
| **`400 Bad Request`** | Input is empty or `< 15` characters | `{"detail": "Input is too short to be a valid resume."}` |
| **`422 Unprocessable Entity`** | LLM returned unparseable non-JSON data | `{"detail": "Unable to extract structured resume data from input."}` |
| **`503 Service Unavailable`** | LLM timeout or rate limits exhausted after 3 attempts | `{"detail": "Resume analysis service unavailable: ..."}` |

---

### Endpoint 2: Stream Resume Analysis (Real-time)
* **Route**: `POST /analyze-resume/stream`
* **Header**: `Content-Type: application/json`
* **Response Content-Type**: `text/event-stream`
* **Behavior**: Streams raw LLM completion tokens chunk-by-chunk for UI typing animations.

---

## 4. Reliability & Production Behaviors

### A. Timeout Strategy
* **Client-level timeout**: Set strictly to **`15.0 seconds`**.
* **Rationale**: Prevents indefinite thread hanging during network degradation or provider cold-starts.

### B. Exponential Backoff Retries
Transient failures are automatically retried up to **3 attempts**:
$$\text{Wait Time} = 2^{(\text{attempt} - 1)} \text{ seconds} \quad (1\text{s} \to 2\text{s} \to 4\text{s})$$

| Failure Type | HTTP Code / Error | Behavior |
| :--- | :--- | :--- |
| **Rate Limit** | HTTP `429` | Wait with backoff and retry |
| **Server Error** | HTTP `500, 502, 503, 504` | Wait with backoff and retry |
| **Network Drop** | `APIConnectionError` | Wait with backoff and retry |
| **Request Timeout**| `APITimeoutError` | Wait with backoff and retry |
| **Invalid Auth / Bad Request** | HTTP `401, 403, 404` | **Fail immediately** (Non-retryable) |

### C. Defensive Schema Design
Messy real-world resumes frequently omit key data (e.g. students have 0 experience; some omit emails). To guarantee the API never throws an unhandled `ValidationError`, fields use fallback defaults:
```python
class ChatResponse(BaseModel):
    name: str = "Unknown"
    email: Optional[str] = None
    skills: list[str] = []
    years_experience: int = 0
```

---

## 5. Verification & Testing

Run the automated test suite verifying edge cases, 5 diverse resumes, and the streaming endpoint:

```bash
python test_resumes.py
```

### Verified Test Cases:
1. **Edge Case 1 (Empty Input)**: Asserts HTTP 400 with clean error detail.
2. **Edge Case 2 (Whitespace Input)**: Asserts HTTP 400 rejection.
3. **Edge Case 3 (Garbled Input `< 15` chars)**: Asserts HTTP 400 rejection.
4. **Resume 1 (Standard Backend)**: Asserts valid extraction with `years_experience: 2`.
5. **Resume 2 (Production Intern)**: Asserts valid extraction on multi-paragraph text.
6. **Resume 3 (Senior Cloud Architect)**: Asserts 12 years experience parsed as `int`.
7. **Resume 4 (Frontend Specialist)**: Asserts list of frontend skills.
8. **Resume 5 (Student Fresher)**: Asserts graceful zero years `years_experience: 0`.
9. **Streaming Endpoint**: Asserts active token streaming via `text/event-stream`.

---

## 6. Calling the API (Copy-Paste Ready Examples)

### cURL
```bash
curl -X POST "http://127.0.0.1:8000/analyze-resume" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Gannu Usha Rani, Software Engineer with 1 year experience in FastAPI, PostgreSQL, and Docker. Contact: usha@example.com"
     }'
```

### Python (`requests`)
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/analyze-resume",
    json={
        "prompt": "John Doe, 3 years exp in Python, Docker. Email: john@doe.com"
    }
)

print("Status:", response.status_code)
print("Data:", response.json())
```
