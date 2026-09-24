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


#Create a prompt template with multiple variables
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a beginner-friendly AI tutor."
        ),
        (
            "human",
            """
Teach me about {topic}.

My learning level is: {level}

My question is:
{question}

Explain the answer clearly and give a simple example.
"""
        )
    ]
)


# Create a chain
chain = prompt | model


# Get multiple inputs from the user
topic = input("Enter the topic: ")
level = input("Enter your level (beginner/intermediate): ")
question = input("Enter your question: ")


# Pass all variables to the chain
response = chain.invoke(
    {
        "topic": topic,
        "level": level,
        "question": question
    }
)


# Display the response
print("\nGemini:")
print(response.text)

