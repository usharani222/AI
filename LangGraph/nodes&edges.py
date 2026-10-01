from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    user_input: str
    category: str
    processed_data: str
    response: str
    execution_trace: list[str]


# Node 1: Get user input
def get_user_input(state: GraphState):
    message = input("Enter your input: ")

    return {
        "user_input": message,
        "execution_trace": state["execution_trace"] + ["get_user_input"]
    }


# Node 2: Classify
def classify(state: GraphState):
    message = state["user_input"].lower()

    if "error" in message or "bug" in message:
        category = "technical"
    else:
        category = "general"

    return {
        "category": category,
        "execution_trace": state["execution_trace"] + ["classify"]
    }


# Node 3: Process
def process(state: GraphState):
    category = state["category"]

    if category == "technical":
        result = "Technical issue identified"
    else:
        result = "General request identified"

    return {
        "processed_data": result,
        "execution_trace": state["execution_trace"] + ["process"]
    }


# Node 4: Respond
def respond(state: GraphState):
    response = (
        f"Category: {state['category']}. "
        f"Result: {state['processed_data']}."
    )

    return {
        "response": response,
        "execution_trace": state["execution_trace"] + ["respond"]
    }


# Build graph
graph = StateGraph(GraphState)

graph.add_node("get_user_input", get_user_input)
graph.add_node("classify", classify)
graph.add_node("process", process)
graph.add_node("respond", respond)


# Define execution flow
graph.add_edge(START, "get_user_input")
graph.add_edge("get_user_input", "classify")
graph.add_edge("classify", "process")
graph.add_edge("process", "respond")
graph.add_edge("respond", END)


# Compile
app = graph.compile()


# Execute
print("Starting the application...")

result = app.invoke({
    "user_input": "",
    "category": "",
    "processed_data": "",
    "response": "",
    "execution_trace": []
})

print("\nFinal Response:", result["response"])
print("Execution Order:", result["execution_trace"])