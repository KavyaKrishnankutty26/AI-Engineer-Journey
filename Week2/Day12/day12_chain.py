import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Make Windows console output use UTF-8
sys.stdout.reconfigure(encoding="utf-8")


# Load API key
load_dotenv()


# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a beginner-friendly Python tutor."
        ),
        (
            "human",
            "Explain {topic} in simple language."
        )
    ]
)


#Connect the prompt and model into a chain
chain = prompt | model


# Get topic from the user
topic = input("Enter a Python topic: ")


# Run the complete chain
response = chain.invoke(
    {
        "topic": topic
    }
)


# Display the response
print("\nGemini:")
print(response.text)

