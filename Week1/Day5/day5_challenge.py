
response = {
    "status": "success",
    "question": "What is AI?",
    "answer": "AI allows computers to perform tasks that normally require human intelligence.",
    "model": "gemini"
}
try:
      
      if(response["status"]) =="success":
       print("Answer:",response["answer"])
      #print(response["name"])
      else:
       print("AI request failed.")

except KeyError:
       print("Missing key") 