from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("API key is loaded",api_key is not None)

client = genai.Client(api_key=api_key)

while True:
 
 question = input("Ask Gemini:")

 if question.lower =="exit":
   break
try:
    interaction = client.interactions.create(
    model = "gemini-3.7-flash",
    input = question
 )
    print(interaction.output_text)
except:
      print("Something went wrong while contacting Gemini.")
      print("Chatbot stopped,Good bye!")
