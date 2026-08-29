#file=open("notes.txt","r")
#content=file.read()
#print(content)
#file.close()

#with open("notes.txt","r") as file:
 #content=file.read()
#print(content)

#with open("test_file.txt","r") as file:
    #content=file.read()
    #print(content)

#with open("ai_notes.txt","w") as file:
    #file.write("Python is my foundation.\n"
#"I am becoming an AI Engineer soon.")

#with open("ai_notes.txt","a") as file:
    #file.write("\nI am learning everyday")
    #file.write("\nNext goal: Build an AI application.")

#with open("ai_notes.txt","r") as file:
    #for line in file:
     #print(line)

with open("ai_notes.txt","r") as file:
    for line in file:
        if "AI" in line:
            print(line)