from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2]/".env"
#print("looking for env at",env_path)
#print("env exists",env_path.exists())
load_dotenv(env_path)

api_key=os.getenv("GEMINI_API_KEY")

client=genai.Client(api_key=api_key)
print("Gemini client created successfully")

#print("API key loaded",api_key is not None)
#print("API key empty",not bool(api_key))
previous_interaction_id = None

while True:

    question = input("You:")
    if question.lower() == "exit":
           print("Goodbye!")
           break
 
    try:
        interaction = client.interactions.create(
          model = "gemini-3.6-flash",
          input = question,
          instruction="You are a beginner-friendly Python tutor. Explain concepts simply.",
          previous_interaction_id=previous_interaction_id
)
        answer = interaction.output_text
        print("Gemini:",answer)
        previous_interaction_id = interaction.id

    except Exception as e:
       print("Something went wrong")




