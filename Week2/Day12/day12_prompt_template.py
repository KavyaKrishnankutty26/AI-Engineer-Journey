import sys

sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

#  Import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate


# Load API key
load_dotenv()


# Create Gemini model through LangChain
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# Create a reusable prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a beginner-friendly Python tutor."
        ),
        (
            "human",
            #"Explain {topic} in simple language."
            #"Explain {topic} with a simple example."
            "Explain {topic} in simple language. Give one example and mention one common mistake beginners make."
        )
    ]
)


# Get topic from the user
topic = input("Enter a Python topic: ")


# Fill the {topic} variable
messages = prompt.invoke(
    {
        "topic": topic
    }
)


# Send the generated messages to Gemini
response = model.invoke(messages)


# Display the AI response
print("\nGemini:")
print(response.text)
