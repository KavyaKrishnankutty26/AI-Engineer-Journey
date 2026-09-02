import json
user = {
    "name": "Kavya",
    "role":"AI Engineer"
}
json_data = json.dumps(user)
print(json_data)

python_data = json.loads(json_data)
print(python_data["name"])
print(python_data["role"])