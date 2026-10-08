import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import ToolMessage


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. MODEL
# ============================================================
from os import getenv
llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
    temperature=0
)


# ============================================================
# 3. TOOL SCHEMAS
# ============================================================

class RefundLookupInput(BaseModel):
    order_id: int = Field(
        description="A unique ID for the order"
    )


class RefundAmountInput(BaseModel):
    order_id: int = Field(
        description="A unique ID for the order"
    )

    refund_percentage: float = Field(
        description="Percentage of the order amount to refund, between 0 and 100",
        ge=0,
        le=100
    )


class SendNotificationInput(BaseModel):
    customer_id: int = Field(
        description="A unique ID for the customer"
    )

    message: str = Field(
        description="The notification message to send"
    )


class SlowRefundInput(BaseModel):
    order_id: int = Field(
        description="A unique ID for the order"
    )


# ============================================================
# 4. TOOLS
# ============================================================

@tool(args_schema=RefundLookupInput)
def refund_lookup(order_id: int) -> str:
    """Check the current status and details of an order."""

    return (
        f"Order {order_id} was delivered "
        f"and the order amount is ₹1200."
    )


@tool(args_schema=RefundAmountInput)
def refund_amount(
    order_id: int,
    refund_percentage: float
) -> str:
    """
    Calculate the refund amount for an order.
    This tool only calculates the refund.
    It does not issue the refund.
    """

    order_amount = 1200

    refund = order_amount * refund_percentage / 100

    return (
        f"Refund amount for order {order_id} "
        f"at {refund_percentage}% is ₹{refund}"
    )


@tool(args_schema=SendNotificationInput)
def send_notification(
    customer_id: int,
    message: str
) -> str:
    """Send a notification message to a customer."""

    return (
        f"Notification sent to customer {customer_id}: "
        f"{message}"
    )


@tool(args_schema=SlowRefundInput)
def slow_refund_service(order_id: int) -> str:
    """Simulates a refund service that takes too long."""

    print("⏳ Slow refund service started...")

    time.sleep(10)

    return f"Refund service completed for order {order_id}"


# ============================================================
# 5. BIND TOOLS
# ============================================================

tools = [refund_lookup,refund_amount,send_notification,slow_refund_service]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 6. TOOL LOOKUP
# ============================================================

tool_map = {
    "refund_lookup": refund_lookup,
    "refund_amount": refund_amount,
    "send_notification": send_notification,
    "slow_refund_service": slow_refund_service
}


# ============================================================
# 7. TIMEOUT WRAPPER
# ============================================================

def run_with_timeout(tool_function, args, timeout=3):

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(
        tool_function.invoke,
        args
    )

    try:

        result = future.result(
            timeout=timeout
        )

        executor.shutdown(wait=False)

        return result

    except FuturesTimeoutError:

        executor.shutdown(wait=False)

        raise TimeoutError(
            f"Tool timed out after {timeout} seconds"
        )


# ============================================================
# 8. EXECUTE TOOL WITH RETRY
# ============================================================

def execute_tool_with_retry(
    tool_name,
    tool_args,
    max_retries=1
):

    tool_function = tool_map[tool_name]

    for attempt in range(max_retries + 1):

        try:

            print(
                f"\n🔧 Tool attempt {attempt + 1}"
            )

            # ------------------------------------------------
            # Timeout tool
            # ------------------------------------------------

            if tool_name == "slow_refund_service":

                result = run_with_timeout(
                    tool_function,
                    tool_args,
                    timeout=3
                )

            # ------------------------------------------------
            # Normal tools
            # ------------------------------------------------

            else:

                result = tool_function.invoke(
                    tool_args
                )

            print("✅ Tool succeeded")

            return {
                "success": True,
                "result": result
            }

        except ValidationError as e:

            print("❌ Invalid tool arguments")

            # Bad arguments usually shouldn't be blindly retried.
            return {
                "success": False,
                "error": (
                    "The tool received invalid arguments. "
                    f"Validation error: {str(e)}"
                )
            }

        except TimeoutError as e:

            print(
                f"⏱️ Timeout on attempt {attempt + 1}"
            )

            if attempt < max_retries:

                print("🔄 Retrying...")

                continue

            return {
                "success": False,
                "error": (
                    "The tool timed out even after "
                    "a retry. The operation was not completed."
                )
            }

        except Exception as e:

            print(
                f"❌ Unexpected tool failure: {e}"
            )

            if attempt < max_retries:

                print("🔄 Retrying...")

                continue

            return {
                "success": False,
                "error": (
                    "The tool failed even after a retry. "
                    "The operation was not completed."
                )
            }


# ============================================================
# 9. CONVERSATION MEMORY
# ============================================================

messages = []


# ============================================================
# 10. AGENT LOOP
# ============================================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent stopped.")
        break

    # --------------------------------------------------------
    # Add user message to conversation memory
    # --------------------------------------------------------

    messages.append(
        ("user", user_input)
    )

    step = 0

    # Maximum number of agent iterations
    max_steps = 10

    while step < max_steps:

        step += 1

        print("\n" + "=" * 50)
        print(f"STEP {step}: MODEL")
        print("=" * 50)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        response = llm_with_tools.invoke(
            messages
        )

        # Save model response to memory
        messages.append(response)

        # ----------------------------------------------------
        # CHECK WHETHER MODEL WANTS TO USE A TOOL
        # ----------------------------------------------------

        if not response.tool_calls:

            print("\nDecision: STOP")

            print(
                f"Final response: {response.content}"
            )

            break

        # ----------------------------------------------------
        # MODEL CHOSE TOOL
        # ----------------------------------------------------

        print("\nDecision: USE TOOL")

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call["args"]

            tool_call_id = tool_call["id"]

            print(
                f"\nTool selected: {tool_name}"
            )

            print(
                f"Arguments: {tool_args}"
            )

            # ------------------------------------------------
            # UNKNOWN TOOL
            # ------------------------------------------------

            if tool_name not in tool_map:

                result = {
                    "success": False,
                    "error": (
                        f"Unknown tool: {tool_name}"
                    )
                }

            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------

            else:

                result = execute_tool_with_retry(
                    tool_name,
                    tool_args,
                    max_retries=1
                )

            # ------------------------------------------------
            # TOOL SUCCESS
            # ------------------------------------------------

            if result["success"]:

                tool_result = result["result"]

                print(
                    f"Tool result: {tool_result}"
                )

                tool_message_content = str(
                    tool_result
                )

            # ------------------------------------------------
            # TOOL FAILURE
            # ------------------------------------------------

            else:

                print(
                    f"Tool failure: {result['error']}"
                )

                tool_message_content = (
                    "IMPORTANT: The tool failed. "
                    "Do not claim that the requested "
                    "operation succeeded.\n\n"
                    f"Failure details: {result['error']}"
                )

            # ------------------------------------------------
            # SEND TOOL RESULT/FAILURE BACK TO MODEL
            # ------------------------------------------------

            messages.append(
                ToolMessage(
                    content=tool_message_content,
                    tool_call_id=tool_call_id
                )
            )

    # --------------------------------------------------------
    # MAX STEP PROTECTION
    # --------------------------------------------------------

    if step >= max_steps:

        print(
            "\n⚠️ Agent stopped because the maximum "
            "number of steps was reached."
        )

        messages.append(
            (
                "system",
                "The agent reached the maximum number "
                "of steps. Do not continue this request."
            )
        )