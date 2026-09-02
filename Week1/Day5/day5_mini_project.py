response = {
    "status": "success",
    "question": "What is Python?",
    "answer": "Python is a programming language.",
    "model": "gemini"
}

try:
    if response["status"] == "success":
        print("Question:", response["question"])
        print("Answer:", response["answer"])
        print("Model:", response["model"])
        print(response["username"])
    else:
        print("The AI request was not successful.")

except KeyError:
    print("Some information is missing from the response.")