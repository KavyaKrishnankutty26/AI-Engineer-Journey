import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

class TextAnalysis(BaseModel):
    topic:str
    sentiment:str
    summary:str

structured_model = model.with_structured_output(
        TextAnalysis
    )

prompt = ChatPromptTemplate.from_messages(
    

    [
        (
            "human",
             """
Analyze the following Text.

Identify:
1.The main topic
2.The sentiment
3.A short summary

Text:
{text}
"""
 )
    ]
)
chain = prompt|structured_model

text = input("Enter text to analyze").strip()
result = chain.invoke(
    {
        "text":text
    }
)
print("\nAnalysis:")
print("Topic:",result.topic)
print("Sentiment",result.sentiment)
print("Summary",result.summary)