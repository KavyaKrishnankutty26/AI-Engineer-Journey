import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# make Windows console output use UTF-8
sys.stdout.reconfigure(encoding="utf-8")

#load API key
load_dotenv()

#create Gemini model
model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

#create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (   "system",
            "You are a beginner-friendly AI tutor"

        ),
        (
            "human",
             """
Teach me about {topic}.
My learning level is: {level}
My question is: {question}

Explain the answer clearly and give a simple example
"""
        )
    ]
)

#create output parser
parser=StrOutputParser()

#connect prompt-> model-> parser
chain = prompt|model|parser

#Reusable AI function
def ask_ai(topic,level,question):

    response = chain.invoke(
        {
            "topic" : topic,
            "level" :level,
            "question" : question
        }
    )

    return response

#main function
def main():

    #get user input
    topic = input("Enter the topic:")
    level= input("Enter your level(beginner/intermediate):")
    question = input("Enter your question:")

#call the reusable ai function

    response = ask_ai(topic, level,question)

#display output
    print("\ngemini")
    print(response)

#run main() only when this file is executed directly
if __name__  == "__main__":
    main()



    