from annotated_types import T
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class TicketState(TypedDict):
    ticket_text:str
    classify:str
    assign:str
    techticket:str
    prodticket:str


def get_ticket_text(state:TicketState):
    text=input("Enter issue:")
    return {"ticket_text":text}

def classify(state:TicketState):
    text=state["ticket_text"].lower()
    if "bug" in text or "issue" in text or "badgateway" in text or "status code" in text:
        category="tech"
    else:
        category="general"
    return {"classify":category}

def assign_ticket(state:TicketState):
    category=state["classify"]
    if category == "tech":
        return "tech_ticket"
    else:
        return "prod_ticket"

def tech_ticket(state:TicketState):
    return {"techticket":"Ticket assigned to teach team"}

def prod_ticket(state:TicketState):
    return {"prodticket":"Ticket assigned to Product team"}


graph=StateGraph(TicketState)

graph.add_node("ticket_text",get_ticket_text)
graph.add_node("category",classify)
graph.add_node("assign",assign_ticket) 
graph.add_node("tech_ticket",tech_ticket)
graph.add_node("prod_ticket",prod_ticket) 

graph.add_edge(START,"ticket_text")
graph.add_edge("ticket_text","category")
graph.add_conditional_edges(
    "category",
    assign_ticket,
    {
        "tech_ticket":"tech_ticket",
        "prod_ticket":"prod_ticket"
    }
)
graph.add_edge("tech_ticket",END)
graph.add_edge("prod_ticket",END)

app=graph.compile()

result=app.invoke({
    "ticket_text":"",
    "classify":"",
    "assign":"",
    "techticket":"",
    "prodticket":""
})
print(result)