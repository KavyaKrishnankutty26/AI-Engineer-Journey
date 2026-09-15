from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel



# Define the expected data structure
class Person(BaseModel):
    name: str
    experience: int
    skills: list[str]


# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Text given to Gemini
text = """
My name is Rahul. I have 4 years of experience
and my skills are Python and AWS.
"""


# Ask Gemini to return structured JSON
interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=text, #sends Rahul's information to Gemini
    response_format={ #I want a structured output
        "type": "text", #response is returned as text
        "mime_type": "application/json", #the text should contain JSON
        "schema": Person.model_json_schema() #takes the Pydantic data and tells what structure the JSON should follow.
    }
)


# Show Gemini's raw JSON response
print("Raw JSON:")
print(interaction.output_text)


# Convert JSON into a Pydantic object
person = Person.model_validate_json(interaction.output_text) #Pydantic check whether the returned data matcches the schema


# Access individual fields
#print("\nParsed data:")
#print("Name:", person.name)
#print("Experience:", person.experience)
#print("Skills:", person.skills)

