
response = {
    "status" :"success",
    "answer" :"python is a programming language",
    "model" :"gemini",
    "usage":{
        "input_tokens":10,
        "output_token":25
    }
}

print(response["status"])
print(response["answer"])
print(response["model"])
print(response["usage"]["input_tokens"])