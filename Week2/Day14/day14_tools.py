import os
import sys
from dotenv import load_dotenv
from langchain_core.tools import tool

# : Import messages needed for tool calling
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key loaded:", bool(api_key))


# --------------------------------------------------
# 2. Create the Gemini model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)


# --------------------------------------------------
# 3. Create a LangChain tool
# --------------------------------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# --------------------------------------------------
# 4. Give the tool to Gemini
# --------------------------------------------------

model_with_tools = model.bind_tools(
    [add_numbers]
)


# --------------------------------------------------
# 5. Test the tool locally
# --------------------------------------------------

local_result = add_numbers.invoke(
    {
        "a": 10,
        "b": 20
    }
)

print("\nLocal tool result:")
print(local_result)


# --------------------------------------------------
# 6.  Create the user's message
# --------------------------------------------------

messages = [
    HumanMessage(
        content="What is 25 + 17? Use the calculator tool."
    )
]


# --------------------------------------------------
# 7.  Ask Gemini to use the tool
# --------------------------------------------------

try:

    response = model_with_tools.invoke(
        messages
    )

    print("\nGemini response:")
    print(response)

    print("\nTool calls:")
    print(response.tool_calls)


    # --------------------------------------------------
    # 8.  Add Gemini's response to the conversation
    # --------------------------------------------------

    messages.append(response)


    # --------------------------------------------------
    # 9. Check if Gemini requested a tool
    # --------------------------------------------------

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        print("\nRequested tool:")
        print(tool_call["name"])

        print("\nTool arguments:")
        print(tool_call["args"])


        # --------------------------------------------------
        # 10. Execute the requested tool
        # --------------------------------------------------

        tool_result = add_numbers.invoke(
            tool_call["args"]
        )

        print("\nTool result:")
        print(tool_result)


        # --------------------------------------------------
        # 11.  Create a ToolMessage
        # --------------------------------------------------

        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )

        messages.append(tool_message)


        # --------------------------------------------------
        # 12. Send the tool result back to Gemini
        # --------------------------------------------------

        final_response = model_with_tools.invoke(
            messages
        )

        print("\nFinal Gemini response:")
        print(final_response.content)


    else:

        print("\nGemini did not request a tool.")


except Exception as e:

    print("\nSomething went wrong.")
    print("Error:", e)

