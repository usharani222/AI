import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class Event(BaseModel):
    name: str = Field(description="Name of the event")
    location: str = Field(description="Location of the event")
    capacity: int = Field(description="Maximum number of attendees")


structured_model = model.with_structured_output(Event)

response = structured_model.invoke(
    """
    We are organizing a Python workshop called
    "Python for Beginners" in Visakhapatnam.
    The venue can accommodate 150 participants.
    """
)

print(response)

print("\nEvent name:", response.name)
print("Location:", response.location)
print("Capacity:", response.capacity)