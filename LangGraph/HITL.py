"""
DAY 48 - LangGraph Human-in-the-Loop
=====================================

Scenario:
    We have a refund system.

    - Refund <= $50  -> automatically processed
    - Refund > $50   -> requires human approval

The graph will PAUSE at an interrupt when human approval is required.

Flow:

START
  |
  v
check_refund
  |
  |---- amount <= 50 ----> process_refund ----> END
  |
  |---- amount > 50 -----> human_approval
                                |
                                | interrupt()
                                |
                                v
                           Human decision
                           /           \
                       approve        reject
                          |              |
                          v              v
                    process_refund      END
                          |
                          v
                         END
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# 1. STATE
# ============================================================

class RefundState(TypedDict):
    customer: str
    amount: float
    requires_approval: bool
    approval: str
    status: str


# ============================================================
# 2. CHECK REFUND NODE
# ============================================================

def check_refund(state: RefundState):
    """
    Check whether the refund is large enough
    to require human approval.
    """

    amount = state["amount"]

    print(f"\n[Check Refund] Refund amount: ${amount}")

    if amount > 50:
        print("[Check Refund] Amount is greater than $50.")
        print("[Check Refund] Human approval is required.")

        return {
            "requires_approval": True
        }

    else:
        print("[Check Refund] Amount is $50 or less.")
        print("[Check Refund] No human approval required.")

        return {
            "requires_approval": False,
            "approval": "approved"
        }


# ============================================================
# 3. ROUTER
# ============================================================

def route_refund(state: RefundState):
    """
    Decide which path the graph should take.

    If approval is required:
        -> go to human_approval

    Otherwise:
        -> directly process the refund
    """

    if state["requires_approval"]:
        return "human_approval"

    return "process_refund"


# ============================================================
# 4. HUMAN APPROVAL NODE
# ============================================================

def human_approval(state: RefundState):
    """
    Pause the graph and ask a human for approval.

    IMPORTANT:
        interrupt() pauses the graph here.

        The graph will NOT continue until
        an external value is provided.
    """

    print("\n[Human Approval] Waiting for human decision...")

    decision = interrupt(
        {
            "message": "Refund requires approval.",
            "customer": state["customer"],
            "amount": state["amount"],
            "question": "Approve or reject this refund?"
        }
    )

    # When the graph resumes,
    # 'decision' will contain the human's answer.

    print(f"[Human Approval] Human decision: {decision}")

    return {
        "approval": decision
    }


# ============================================================
# 5. ROUTE HUMAN DECISION
# ============================================================

def route_after_approval(state: RefundState):
    """
    After the human gives a decision:

        approve -> process refund

        reject  -> end workflow
    """

    if state["approval"].lower() == "approve":
        return "process"

    return "reject"


# ============================================================
# 6. PROCESS REFUND NODE
# ============================================================

def process_refund(state: RefundState):
    """
    Actually process the refund.

    This node runs ONLY after:
        - automatic approval for small refunds
        OR
        - human approval for large refunds.
    """

    print(
        f"\n[Process Refund] Processing ${state['amount']} "
        f"refund for {state['customer']}."
    )

    return {
        "status": "refund_processed"
    }


# ============================================================
# 7. BUILD GRAPH
# ============================================================

builder = StateGraph(RefundState)


# Register nodes
builder.add_node("check_refund", check_refund)
builder.add_node("human_approval", human_approval)
builder.add_node("process_refund", process_refund)


# ------------------------------------------------------------
# START -> check_refund
# ------------------------------------------------------------

builder.add_edge(
    START,
    "check_refund"
)


# ------------------------------------------------------------
# check_refund -> conditional routing
# ------------------------------------------------------------

builder.add_conditional_edges(
    "check_refund",
    route_refund,
    {
        "human_approval": "human_approval",
        "process_refund": "process_refund"
    }
)


# ------------------------------------------------------------
# human_approval -> conditional routing
# ------------------------------------------------------------

builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "process": "process_refund",
        "reject": END
    }
)


# ------------------------------------------------------------
# process_refund -> END
# ------------------------------------------------------------

builder.add_edge(
    "process_refund",
    END
)


# ============================================================
# 8. CHECKPOINTER
# ============================================================

# The checkpointer saves the graph state.

checkpointer = InMemorySaver()


# Compile the graph with persistence
app = builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 9. THREAD ID
# ============================================================

# thread_id identifies this particular workflow.

config = {
    "configurable": {
        "thread_id": "refund-123"
    }
}


# ============================================================
# 10. INITIAL STATE
# ============================================================

initial_state = {
    "customer": "Usha",
    "amount": 10,
    "requires_approval": False,
    "approval": "",
    "status": "pending"
}


# ============================================================
# 11. FIRST RUN
# ============================================================

print("====================================")
print("STARTING REFUND WORKFLOW")
print("====================================")

result = app.invoke(
    initial_state,
    config=config
)

print("\nGraph execution returned:")
print(result)