from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from api import llmcall, ChatResponse

app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat", response_model=ChatResponse)
def chatbot(request: ChatRequest):

    result = llmcall(request.prompt)

    return result


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )