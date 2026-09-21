import os
from dotenv import load_dotenv
#Import Gemini  Langchain integration
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv() #load api key

api_key = os.getenv("GEMINI_API_KEY")

#create Gemini model through Langchain
model = ChatGoogleGenerativeAI( 
    model = "gemini-3.6-flash",
)
#send a message
response = model.invoke("What is python? Explain in one sentence.")

#display the response
print(response.text)




