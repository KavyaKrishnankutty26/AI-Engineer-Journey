from pydantic import BaseModel

class Person(BaseModel):
    name:str
    experience:int
    skills:list[str]

person = Person(
        name = "Rahul",
        experience = 4,
        skills = ["python","AWS"]
    )
print(person)
print(person.name)
print(person.experience)
print(person.skills)


    