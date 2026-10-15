
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

# Create Gemini model through LangChain
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


#Import message types
from langchain_core.messages import SystemMessage, HumanMessage


#Create a system message
system_message = SystemMessage(
    content="You are a beginner-friendly Python tutor."
)


#Create a human message
human_message = HumanMessage(
    content="What is a Python list?"
)


#Send multiple messages to Gemini
response = model.invoke(
    [
        system_message,
        human_message
    ]
)


# Display the AI response
print(response.text)

