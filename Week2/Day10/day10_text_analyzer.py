from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
from typing import Literal


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Define the expected AI response structure
# --------------------------------------------------

class TextAnalysis(BaseModel):
    topic: str
    sentiment: Literal["positive", "negative", "neutral"]
    summary: str


# --------------------------------------------------
# 3. Analyze text using Gemini
# --------------------------------------------------

def analyze_text(text):

    try:

        # Create a reliable prompt
        prompt = f"""
Analyze the following text.

Identify:
1. The main topic.
2. The sentiment.
3. A short summary.

Rules:
- Analyze only the text provided.
- Keep the summary concise.
- Use only one of these sentiment values: positive, negative, or neutral.
- If the text does not contain enough meaningful information, do not guess or invent an analysis.
- If specific information is not provided in the text, describe it as unknown instead of guessing.

Text:
{text}
"""

        # Send request to Gemini
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": TextAnalysis.model_json_schema()
            }
        )

        # Show raw JSON returned by Gemini
        print("\nRaw JSON:")
        print(interaction.output_text)

        # Validate and convert JSON into a Pydantic object
        analysis = TextAnalysis.model_validate_json(
            interaction.output_text
        )

        return analysis

    except Exception as e:

        print("\nSomething went wrong.")
        print("Error:", e)

        return None


# --------------------------------------------------
# 4. Main application
# --------------------------------------------------

def main():

    # Get input from user
    text = input("Enter text to analyze: ")

    # Clean input
    text = text.strip()

    # Validate input
    if not text:
        print("Please enter some text.")
        return

    # Analyze the text
    analysis = analyze_text(text)

    # Display result
    if analysis:

        print("\nParsed analysis:")
        print("Topic:", analysis.topic)
        print("Sentiment:", analysis.sentiment)
        print("Summary:", analysis.summary)

    else:

        print("\nThe text could not be analyzed.")


# --------------------------------------------------
# 5. Start the application
# --------------------------------------------------

if __name__ == "__main__":
    main()
