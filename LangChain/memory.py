import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
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
# HELPER FUNCTION
# ============================================================

def print_memory(title, memory):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(memory.load_memory_variables({}))


# ============================================================
# 1. CONVERSATION BUFFER MEMORY
# ============================================================

print("\n\n")
print("=" * 80)
print("1. CONVERSATION BUFFER MEMORY")
print("=" * 80)

buffer_memory = ConversationBufferMemory()

buffer_memory.save_context(
    {"input": "My name is Usha."},
    {"output": "Nice to meet you, Usha!"}
)

buffer_memory.save_context(
    {"input": "I am learning LangChain."},
    {"output": "That's great!"}
)

buffer_memory.save_context(
    {"input": "I want to learn RAG next."},
    {"output": "RAG is an important next step."}
)

print_memory(
    "Complete conversation stored in Buffer Memory",
    buffer_memory
)


# ============================================================
# 2. CONVERSATION BUFFER WINDOW MEMORY
# ============================================================

print("\n\n")
print("=" * 80)
print("2. CONVERSATION BUFFER WINDOW MEMORY")
print("=" * 80)

window_memory = ConversationBufferWindowMemory(
    k=2
)

window_memory.save_context(
    {"input": "My name is Usha."},
    {"output": "Nice to meet you!"}
)

window_memory.save_context(
    {"input": "I am learning Python."},
    {"output": "Python is useful for AI."}
)

window_memory.save_context(
    {"input": "I am learning LangChain."},
    {"output": "LangChain is useful for LLM applications."}
)

window_memory.save_context(
    {"input": "I want to learn RAG."},
    {"output": "RAG is used to connect LLMs with external knowledge."}
)

print_memory(
    "Only recent conversation is retained",
    window_memory
)


# ============================================================
# 3. CONVERSATION SUMMARY MEMORY
# ============================================================

print("\n\n")
print("=" * 80)
print("3. CONVERSATION SUMMARY MEMORY")
print("=" * 80)

summary_memory = ConversationSummaryMemory(
    llm=model
)

summary_memory.save_context(
    {"input": "My name is Usha."},
    {"output": "Nice to meet you, Usha!"}
)

summary_memory.save_context(
    {"input": "I am learning Python and LangChain."},
    {"output": "Both are useful for building AI applications."}
)

summary_memory.save_context(
    {"input": "I want to learn RAG and then agents."},
    {"output": "That is a common progression for LLM application development."}
)

print_memory(
    "Conversation compressed into a summary",
    summary_memory
)


# ============================================================
# 4. COMPARE ALL THREE
# ============================================================

print("\n\n")
print("=" * 80)
print("4. COMPARISON")
print("=" * 80)

print("""
ConversationBufferMemory
------------------------
Stores the complete conversation.

ConversationBufferWindowMemory
------------------------------
Stores only the recent window of conversation.

ConversationSummaryMemory
--------------------------
Continuously summarizes the conversation.
""")


# ============================================================
# 5. SIMPLE CHATBOT WITH BUFFER MEMORY
# ============================================================

print("\n\n")
print("=" * 80)
print("5. CHATBOT WITH MEMORY")
print("=" * 80)

chat_memory = ConversationBufferMemory(
    return_messages=True
)


def chat(user_input):

    # Get previous conversation
    history = chat_memory.load_memory_variables({})["history"]

    # Create messages for the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI tutor. "
                "Use the conversation history when answering."
            )
        }
    ]

    # Add previous messages
    for message in history:
        messages.append({
            "role": "user" if message.type == "human" else "assistant",
            "content": message.content
        })

    # Add current question
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Call model
    response = model.invoke(messages)

    # Save conversation
    chat_memory.save_context(
        {"input": user_input},
        {"output": response.content}
    )

    return response.content


# Conversation 1
response = chat("My name is Usha.")

print("\nUser:", "My name is Usha.")
print("AI:", response)


# Conversation 2
response = chat("I am learning LangChain.")

print("\nUser:", "I am learning LangChain.")
print("AI:", response)


# Conversation 3
response = chat("What am I learning?")

print("\nUser:", "What am I learning?")
print("AI:", response)


# Show stored memory
print("\nStored memory:")
print(chat_memory.load_memory_variables({}))