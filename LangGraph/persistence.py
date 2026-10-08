import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ==========================================
# 1. State
# ==========================================
class TicketState(TypedDict):
    ticket: str
    analysis: str
    status: str


# ==========================================
# 2. Nodes
# ==========================================
def analyze_ticket(state: TicketState):
    print("\n[Node 1: analyze] Analyzing ticket...")
    return {
        "analysis": f"Issue verified: {state['ticket']}",
        "status": "analyzed"
    }


def process_ticket(state: TicketState):
    print("\n[Node 2: process] Processing ticket resolution...")
    return {
        "status": "completed"
    }


# ==========================================
# 3. Build Graph
# ==========================================
builder = StateGraph(TicketState)
builder.add_node("analyze", analyze_ticket)
builder.add_node("process", process_ticket)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "process")
builder.add_edge("process", END)


# ==========================================
# 4. Checkpointer & Breakpoint
# ==========================================
conn = sqlite3.connect("tickets.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Pause execution automatically before 'process' node
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["process"]
)

config = {"configurable": {"thread_id": "ticket-101"}}


# ==========================================
# 5. Start or Resume Automatically
# ==========================================
if __name__ == "__main__":
    snapshot = app.get_state(config)

    if not snapshot.values:
        # Run 1: First time run - starts fresh and pauses before 'process'
        print("--- [Run 1: Starting fresh workflow] ---")
        initial_state = {
            "ticket": "Customer requests a refund",
            "analysis": "",
            "status": "new"
        }
        app.invoke(initial_state, config=config)

        print("\n[PAUSED] Paused at checkpoint before 'process'.")
        print("         State saved to tickets.db.")
        print("         Run 'python persistence.py' again to resume from this checkpoint!")

    elif snapshot.next:
        # Run 2: Found saved checkpoint - resumes where it left off
        print("--- [Run 2: Resuming from checkpoint] ---")
        print(f"Next node to execute: {snapshot.next}")

        # Passing None tells LangGraph to continue from the saved checkpoint
        final_state = app.invoke(None, config=config)

        print("\n[COMPLETED] Workflow Finished!")
        print("Final State:", final_state)

    else:
        # Already finished
        print("--- Workflow already completed for this ticket ---")
        print("Final State:", snapshot.values)