import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()

class LearningAnswer(BaseModel):
    topic:str
    explanation:str
    example:str
    key_point:str
    study_tip:str

model = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite"
)
structured_model = model.with_structured_output(
    LearningAnswer
    ) 


prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
"""
You are an AI learning assistant.

Teach the user according to their learning level.

For beginner:
- Use simple language.
- Avoid unnecessary technical terms.
- Explain concepts step by step.
- Give a simple example.

For intermediate:
- Provide more technical detail.
- Explain why the concept works.
- Give a practical example.

For all levels:
- Answer only the user's question.
- Be accurate.
- Keep the explanation focused.
- Make the key point easy to remember.
- Give one practical study tip related to the topic. 
"""
        ),
        (
            "human",
            """
Topic : {topic}
Learning level : {level}
Question : {question}

"""
        )
    ]
)
chain = prompt | structured_model

def ask_ai(topic, level, question):

    try:

        result = chain.invoke(
            {
                "topic": topic,
                "level": level,
                "question": question
            }
        )

        return result

    except Exception as e:

        print("\nSomething went wrong.")
        print("Please try again later.")
        print("Error:", e)

        return None
    

def main():

   topic = input("Enter the topic:").strip()
   level = input("Enter your level (beginner,intermediate:)").strip().lower()
   question = input("Enter your question:").strip()

   if not topic:
    print("please enter a topic")
    return

   if level not in ["beginner","intermediate"]:
    print("Please enter either beginner/intermediate")
    return
 
   if not question:
    print("Please enter a queation")
    return

   result = ask_ai(
  topic,
  level,
  question)

   if result is None:
      print("\nAI could not send an answer")
      return

   print("\n AI Learning Assistant")
   print("Topic:",result.topic)

   print("\nExplanation")
   print(result.explanation)

   print("\nExample:")
   print(result.example)

   print("\nKey-point")
   print(result.key_point)

   print("\nStudy-tip")
   print(result.study_tip)






if __name__ == "__main__":
   main()

