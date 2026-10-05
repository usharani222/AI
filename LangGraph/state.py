from langchain_core.runnables import graph
from numpy import gradient
from pandas.core.indexes import category
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

class SupportTicketState(TypedDict):
    ticket_text:str
    category:str
    priority:str
    resolve:str

def get_ticket(state: SupportTicketState):
    ticket_text=input("Enter the issue faced:")
    return {"ticket_text":ticket_text}

def classify(state: SupportTicketState):
    text=state["ticket_text"].lower()
    if "login in" in text or "sign in" in text:
        category="authentication"
    elif "bug" in text or "error" in text:
        category="technical"
    elif "payment" in text or "money" in text:
        category="billing"
    else:
        category="general"
    return {"category":category}

def set_priority(state: SupportTicketState):
    category= state["category"]
    if category == "authentication":
        priority = "very high"
    elif category == "billing":
        priority = "high"
    elif category == "technical":
        priority = "medium"
    else:
        priority = "low"
    return {"priority":priority}

def resolve(state: SupportTicketState):
    priority=state["priority"]
    if priority == "very high":
        resolve="Logout and login ,change password and clear cache"
    elif priority == "high":
        resolve="Share the billing info via mail also check bankaccount if deducted or not confirm first"
    elif priority == "medium":
        resolve="your concern has been forwarded to technical team"
    else:
        resolve="Please try after some time"
    return {"resolve":resolve}

## graph

graph=StateGraph(SupportTicketState)

graph.add_node("ticket_data",get_ticket)
graph.add_node("classify",classify)
graph.add_node("priority",set_priority)
graph.add_node("resolve",resolve)

graph.add_edge(START,"ticket_data")
graph.add_edge("ticket_data","classify")
graph.add_edge("classify","priority")
graph.add_edge("priority","resolve")
graph.add_edge("resolve",END)

app=graph.compile()
print("Starting the application...\nGraph compiled\n")

result=app.invoke(
    {
        "ticket_text":"",
        "category":"",
        "priority":"",
        "resolve":""
    }
)

print(result["resolve"])
