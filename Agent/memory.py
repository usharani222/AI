## Multiple Tools with Pydantic

## 1. Tool1
from os import getenv
from langchain.tools import tool
from pydantic import BaseModel, Field

class refund_lookup(BaseModel):
    order_id:int = Field(description="This is a unique id for the order")
@tool(args_schema=refund_lookup)
def refund_lookup(order_id:int)->int:
    """This is used to check the order current status and details"""
    return f"order {order_id} delivered and amout is 1200"

## Tool2
class refund_amount(BaseModel):
    order_id:int=Field(description="This a unique id for order")
    refund_percentage:float=Field(description="The percentage of amount to be refunded between 0 to 100",ge=0,le=100)

@tool(args_schema=refund_amount)
def refund_amount(order_id:int,refund_percentage:float)->int:
    """This calculates the amount to be refunded based on the percentage and order id given just calculates and returns the amount to be refunded"""
    amount=50000
    refund_amount=(amount)*refund_percentage/100
    return f"Amount to be refunded for {order_id} is {refund_amount}"

## Tool3
class sendnotification(BaseModel):
    customer_id:int=Field(description="A unique id for customer")
    notification:str=Field(description="The message to be sent as notification to customer")

@tool(args_schema=sendnotification)
def send_notification(customer_id:int,notification:str)->str:
    """This send a notification message to the customer"""
    return f"send to {customer_id}:{notification}"

## model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY")
)

llm_with_tools=llm.bind_tools([refund_lookup,refund_amount,send_notification])

## loop
from langchain.messages import ToolMessage
message=[]
step =0
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # Add the new user message to existing history
    message.append(("user", user_input))
    while True:
        step+=1
        print(f"\n{'=' * 50}")
        print(f"STEP {step}: MODEL")
        print(f"{'=' * 50}")

        response=llm_with_tools.invoke(message)
        message.append(response)
        if not response.tool_calls:
            print(response.content)
            break
        for tool_call in response.tool_calls:
            print(f"Tool call : {tool_call["name"]}\n")
            if tool_call["name"] == "refund_lookup":
                result = refund_lookup.invoke(tool_call["args"])
                message.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
            elif tool_call["name"] == "refund_amount":
                result = refund_amount.invoke(tool_call["args"])
                message.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
            else:
                result = send_notification.invoke(tool_call["args"])
                message.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
            print(f"Tool result: {result}")

            