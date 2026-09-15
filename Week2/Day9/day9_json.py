import json

#person = { 
    #"name" : "Rahul",
    #"age" : 25,
    #"skills" : ["python","AWS"]

#}
json_data = '{"name": "Rahul","age":25,"skills": ["python","AWS"]}'

#json_data = json.dumps(person)
person = json.loads(json_data)

#print(json_data)
#print(type(person))
#print(type(json_data))

print(person)
print(type(person))
print(person["name"])
print(person["skills"])
print(person["age"])