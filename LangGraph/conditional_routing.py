"""
Conditional Routing in LangGraph
================================
This example demonstrates dynamic branching (conditional edges) in LangGraph.
Based on the user's issue text, the graph classifies the ticket as 'tech' or 'general'
and dynamically routes the flow to either the Technical Support team or the Product team.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. State Definition
# ==========================================
# Defines the shared state schema passed across all nodes in the graph.
class TicketState(TypedDict):
    ticket_text: str
    classify: str
    assign: str
    techticket: str
    prodticket: str


# ==========================================
# 2. Node Functions (Action / Task Steps)
# ==========================================

# Step 1: Prompt the user to enter their ticket issue
def get_ticket_text(state: TicketState):
    text = input("Enter issue: ")
    return {"ticket_text": text}


# Step 2: Classify the ticket based on keywords
def classify(state: TicketState):
    text = state["ticket_text"].lower()
    # Check for technical keywords in the issue text
    if any(keyword in text for keyword in ["bug", "issue", "badgateway", "bad gateway", "status code", "500", "error"]):
        category = "tech"
    else:
        category = "general"
    return {"classify": category}


# Step 3a: Destination node for technical tickets
def tech_ticket(state: TicketState):
    return {"techticket": "Ticket assigned to Tech team"}


# Step 3b: Destination node for general/product tickets
def prod_ticket(state: TicketState):
    return {"prodticket": "Ticket assigned to Product team"}


# ==========================================
# 3. Router Function (Conditional Logic)
# ==========================================
# The router function inspects the current state and returns a string key
# that matches the path mapping dictionary in `add_conditional_edges`.
def assign_ticket(state: TicketState) -> str:
    category = state["classify"]
    if category == "tech":
        return "tech_ticket"
    else:
        return "prod_ticket"


# ==========================================
# 4. Graph Construction
# ==========================================
graph = StateGraph(TicketState)

# Register graph nodes
graph.add_node("ticket_text", get_ticket_text)
graph.add_node("category", classify)
graph.add_node("tech_ticket", tech_ticket)
graph.add_node("prod_ticket", prod_ticket)

# Normal edges: Sequential execution from START -> ticket_text -> category
graph.add_edge(START, "ticket_text")
graph.add_edge("ticket_text", "category")

# Conditional edge: Dynamic routing after 'category' node
# Parameters:
#   source: The node after which the decision is made ("category")
#   path: The routing function returning a string key (assign_ticket)
#   path_map: Dictionary mapping returned key to the destination node name
graph.add_conditional_edges(
    "category",
    assign_ticket,
    {
        "tech_ticket": "tech_ticket",
        "prod_ticket": "prod_ticket"
    }
)

# Connect terminal nodes to END
graph.add_edge("tech_ticket", END)
graph.add_edge("prod_ticket", END)


# ==========================================
# 5. Compilation & Execution
# ==========================================
app = graph.compile()

# Invoke the graph with initial empty state
result = app.invoke({
    "ticket_text": "",
    "classify": "",
    "assign": "",
    "techticket": "",
    "prodticket": ""
})

# Display the final state dictionary
print("\nFinal State:")
print(result)