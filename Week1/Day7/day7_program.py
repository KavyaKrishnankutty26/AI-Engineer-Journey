conversation = []

while True:
    question = input("You: ")

    if question == "exit":
        break

    conversation.append({
        "role": "user",
        "content": question
      })

    #print(conversation)
    answer = "This is a  simulated AI response"

    conversation.append({
        "role":"assitant",
        "content":answer
    })
    print("AI:",answer)
print("\nConversation history")

for message in conversation:
    print(message["role"],":", message["content"])
