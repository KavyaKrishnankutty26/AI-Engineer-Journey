#api_response = {
    #"status":"success",
    #"model":"Gemini",
    #"output":"Python is a programming language"
#}
#print(api_response["status"])
#print(api_response["model"])
#print(api_response["output"])

api_response = {
    "status": "success",
    "model": "gemini",
    "question": "What is Python?",
    "answer": "Python is a programming language.",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 25
    }
}

print("Question:", api_response["question"])
print("Answer:", api_response["answer"])
print("Model:", api_response["model"])
print("Input tokens:", api_response["usage"]["input_tokens"])