from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing import TypedDict

class SupportTicketState(TypedDict):
    ticket_id:int
    ticket_text:str
    category:str
    assigned_team:str
    is_valid:bool
    is_sensitive:bool
    attempt_count:int
    approval:str
    status:str

def classification(state:SupportTicketState):
    text = state["ticket_text"].lower()
    if "bug" in text or "issue" in text or "error" in text :
        category = "technical"
    elif "payment" in text or "billing" in text :
        category = "billing"
    else :
        category = "general"
    return {"category":category}

def route(state:SupportTicketState):
    category=state["category"]
    if category == "billing":
        assigned = "support"
    else:
        assigned = "technical"
    return {"assigned_team" : assigned}

def validate(state:SupportTicketState):
    if state["category"] == "billing" and state["assigned_team"] == "support":
        valid = True
    elif state["category"] == "technical" and state["assigned_team"] == "technical":
        valid = True
    elif state["category"] == "general" and state["assigned_team"] == "technical":
        valid = True
    else :
        valid = False
    return {
        "is_valid" : valid
    }

def is_sensitive(state:SupportTicketState):
    if state["is_sensitive"] == True :
        return "Human_approval"
    else :
        return "resolve"
    
def invalid(state:SupportTicketState):
    if state["attempt_count"]<=3:
        print("Retying...")
        return "classify"
    else:
        print("Max retries reached...")
        return "End"

def approval(state:SupportTicketState):
    if state["approval"] != "approved":
        approval = "approved"
    else :
        approval = "approved"
    return {
        "approval":approval
    }

def human_loop(state:SupportTicketState):
    human_response = interrupt("Please approve or reject this ticket:")
    return {"status": human_response}

def check_human_status(state:SupportTicketState):
    if state["status"].lower() == "accept":
        return "accept"
    else:
        return "End"

graph = StateGraph(SupportTicketState)
graph.add_node("classify",classification)
graph.add_node("route",route)
graph.add_node("validate",validate)
graph.add_node("human_loop",human_loop)
graph.add_node("approval",approval)

graph.add_edge(START,"classify")
graph.add_edge("classify","route")
graph.add_edge("route","validate")
graph.add_conditional_edges(
    "validate",
    is_sensitive,
    {
        "Human_approval":"human_loop",
        "resolve":"route"
    }
)
graph.add_conditional_edges(
    "human_loop",
    check_human_status,
    {
        "accept":"approval",
        "End":END
    }
)
graph.add_edge("approval",END)

app=graph.compile(
    checkpointer=InMemorySaver()
)
config = {
    "configurable": {
        "thread_id": "refund-123"
    }
}

initial_state={
        "ticket_id":12345,
        "ticket_text":"payment failed",
        "category":"",
        "assigned_team":"",
        "is_valid":False,
        "is_sensitive":True,
        "attempt_count":0,
        "approval":"",
        "status":""
    }

result = app.invoke(
    initial_state,
    config=config
)

print(result)

