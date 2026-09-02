response = {
    "status": "success",
    "answer": "Python is a programming language."
}

if response["status"] == "error":
    print(response["answer"])
else:
    print("Something went wrong.")