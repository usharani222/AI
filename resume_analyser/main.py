from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from chat import chat, stream_chat, ChatResponse

# Initialize FastAPI application
app = FastAPI(
    title="Resume Analyzer API",
    description="Analyze and extract structured information from resumes with retries, timeout handling, and streaming.",
    version="1.0.0"
)

# Request schema for resume input
class ResumeRequest(BaseModel):
    prompt: str

# Primary endpoint to extract structured resume details
@app.post("/analyze-resume", response_model=ChatResponse)
@app.post("/resume-analizer", response_model=ChatResponse, include_in_schema=False)
def analyze_resume(request: ResumeRequest):
    # Reject empty or whitespace-only input with HTTP 400
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Resume text cannot be empty."
        )

    clean_prompt = request.prompt.strip()
    # Reject garbled or excessively short input with HTTP 400
    if len(clean_prompt) < 15:
        raise HTTPException(
            status_code=400,
            detail="Input is too short to be a valid resume. Please provide at least 15 characters of resume content."
        )

    try:
        # Call chat function and return validated ChatResponse
        return chat(clean_prompt)
    except ValueError as e:
        # Return HTTP 422 if model output cannot be parsed into valid JSON
        raise HTTPException(
            status_code=422,
            detail=f"Unable to extract structured resume data from input: {str(e)}"
        )
    except RuntimeError as e:
        # Return HTTP 503 if API encounters persistent timeouts or rate limits
        raise HTTPException(
            status_code=503,
            detail=f"Resume analysis service unavailable: {str(e)}"
        )

# Secondary endpoint streaming token chunks via server-sent events
@app.post("/analyze-resume/stream")
def analyze_resume_stream(request: ResumeRequest):
    # Reject empty input before initiating stream
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Resume text cannot be empty."
        )

    # Stream completion chunks as they are received from the LLM
    return StreamingResponse(
        stream_chat(request.prompt.strip()),
        media_type="text/event-stream"
    )

# Run server on localhost:8000
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
