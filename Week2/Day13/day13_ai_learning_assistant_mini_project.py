import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a beginner friendly AI-tutor"
        ),
        (
            "human",
             """
Teach me about {topic}.
My learning level is : {level}
My question is: {question}
Explain the answer clearly and give a simple example.
             """
        )
    ]
)

parser = StrOutputParser()
chain = prompt|model|parser

def ask_ai(topic,level,question):
    response = chain.invoke(
        {
            "topic" : topic,
            "level" : level,
            "question": question
        }
    )
    return response

def main():
    topic = input("Enter the topic:").strip()
    if not topic:
        print("please enter a topic")
        return
    level = input("Enter your level beginner/intermediate:").strip().lower()
    if level not in [beginner/intermediate]:
        print("Please enter beginner/intermediate")
        return
    question = input("Enter your question").strip()
    if not question:
        print("Please enter a question")
        return

    try:
        response = ask_ai(topic,
                          level,
                          question
                          )
    except Exception as e:
        print("\nSomething went wrong!")
        print("Please try again later")
        print("Error",e)
        return
    print("\nGemini:")
    print(response)

if __name__ =="__main__":
    main()
