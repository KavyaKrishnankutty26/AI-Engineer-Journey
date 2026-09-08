from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2]/".env"
load_dotenv(env_path)
api_key=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = api_key)
print("Gemini client created successfully")

PROMPT_TEMPLATE = """ 
You are a beginner friendly Python tutor.

Explain the user's question for someone who is learning Python.

Requirements:
-Use simple language
-Give one small example when useful
-Avoid unnecessary advanced terminology
-Keep the answer concise

User question:
{question}
"""

def create_prompt(question):
    return PROMPT_TEMPLATE.format(question=question)

previous_interaction_id = None

while True:
    question = input("\nYou:")

    if question.lower() == "exit":
        print("Goodbye!")

    prompt = create_prompt(question)

    try:
        interaction = client.interactions.create(
            model = "gemini-3.6-flash",
            input=prompt,
            previous_interaction_id=previous_interaction_id

        )

        answer = interaction.output_text
        print("\nGemini:",answer)

        previous_interaction_id=interaction.id

    except Exception as e:
        print("Something went wrong:",e)