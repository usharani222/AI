"""
LangGraph Loops & Retries Workflow
===================================
Demonstrates a loop/retry pattern in LangGraph with a 2-attempt limit:
1. `generate` node: Generates output. On attempt 1, it deliberately produces
   an invalid response. On attempt 2, it produces a valid response.
2. `validate` node: Checks if output meets requirements.
3. `should_retry` router: Loops back to `generate` if invalid and attempts < 2,
   otherwise routes to `END`.
"""

import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. State Definition
# ==========================================
class RetryState(TypedDict):
    prompt: str
    response: str
    attempt_count: int
    is_valid: bool
    feedback: str


# ==========================================
# 2. Node Functions
# ==========================================

def generate_output(state: RetryState):
    """
    Generates output based on the current attempt count.
    Deliberately forces attempt 1 to produce invalid JSON to trigger a retry.
    """
    current_attempt = state.get("attempt_count", 0) + 1
    print(f"\n[Generate Node] Starting Attempt #{current_attempt}...")

    if current_attempt == 1:
        # Deliberately invalid output (broken JSON missing closing bracket)
        raw_output = '{"status": "error", "message": "missing bracket"'
        print("  -> Attempt 1: Intentionally produced INVALID output.")
    else:
        # Valid output on retry
        raw_output = json.dumps({
            "status": "success",
            "message": f"Resolved: '{state['prompt']}' processed successfully."
        })
        print("  -> Attempt 2: Produced VALID output.")

    return {
        "response": raw_output,
        "attempt_count": current_attempt
    }


def validate_output(state: RetryState):
    """
    Validates the generated output (verifies valid JSON).
    """
    print("[Validate Node] Validating response...")
    raw = state["response"]

    try:
        parsed = json.loads(raw)
        print("  -> Validation PASSED!")
        return {
            "is_valid": True,
            "feedback": "Output is valid JSON."
        }
    except json.JSONDecodeError as err:
        print(f"  -> Validation FAILED: {err.msg}")
        return {
            "is_valid": False,
            "feedback": f"Invalid JSON syntax: {err.msg}"
        }


# ==========================================
# 3. Router Function (Loop / Retry Condition)
# ==========================================
def should_retry(state: RetryState) -> str:
    """
    Decides whether to loop back to generate or exit:
    - If valid -> end workflow
    - If invalid and attempts < 2 -> retry (loop back)
    - If invalid and attempts >= 2 -> stop (max retries reached)
    """
    if state["is_valid"]:
        print("[Router] Result is valid -> routing to END.")
        return "end"

    if state["attempt_count"] < 2:
        print("[Router] Invalid output and attempts < 2 -> LOOPING BACK to generate.")
        return "retry"

    print("[Router] Max attempts reached -> routing to END.")
    return "end"


# ==========================================
# 4. Graph Construction
# ==========================================
graph = StateGraph(RetryState)

# Register nodes
graph.add_node("generate", generate_output)
graph.add_node("validate", validate_output)

# Flow: START -> generate -> validate
graph.add_edge(START, "generate")
graph.add_edge("generate", "validate")

# Conditional Edge (Loop / Exit):
# From 'validate', evaluate should_retry:
# - "retry" -> loop back to "generate"
# - "end"   -> route to END
graph.add_conditional_edges(
    "validate",
    should_retry,
    {
        "retry": "generate",
        "end": END
    }
)

# Compile graph
app = graph.compile()


# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    print("=== Starting 2-Attempt Retry Workflow ===")

    initial_state = {
        "prompt": "Process user payment request",
        "response": "",
        "attempt_count": 0,
        "is_valid": False,
        "feedback": ""
    }

    final_state = app.invoke(initial_state)

    print("\n=== Final State Summary ===")
    print(f"Total Attempts : {final_state['attempt_count']}")
    print(f"Is Valid       : {final_state['is_valid']}")
    print(f"Final Response : {final_state['response']}")
    print(f"Feedback       : {final_state['feedback']}")
