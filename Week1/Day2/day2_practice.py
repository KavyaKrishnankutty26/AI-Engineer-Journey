
#age = 25
#if age >=18 :
   #print ("eligible to vote")

#age = 15
#if age >18:
    #print("Adult")
#else:
    #print("Minor")


#marks =85
#if marks>=95:
    #print("Grade A")
#elif marks>=75:
    #print("Grade B")
#elif marks>=65:
    #print("Grade C")
#else:
    #print("Fail")

#skills = ["python","SQL","Git","AI"]
#for skill in skills:
    #if len(skill)>=4:
     #print(skill)
#print(skills[0])
#skills.append("RAG")
#print (skills)

#technologies = ["python","sql","git"]
#print(technologies[0])
#technologies.append("LLM")
#print(technologies)

#person = {
    #"name" :"Kavya",
    #"role" :"AI Engineer",
    #"experience" :2
#}
#print(person)
#print(person["name"])
#print(person["role"])

#job = {
    #"title":"engineer",
    #"location":"dubai",
    #"experience":2
#}
#job["title"]="ai engineer"
#job["skills"]="python"
#print(job)
#print(job["title"])
#print(job["location"])
#print(job["experience"])

jobs = [
    {"title": "AI Engineer", "location": "Dubai"},
    {"title": "Data Engineer", "location": "Dubai"},
    {"title": "Python Developer", "location": "Abu Dhabi"}
]
for job in jobs:
    print(job["title"])