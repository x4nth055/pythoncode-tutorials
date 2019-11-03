import pickle

class Person:
    def __init__(self, first_name, last_name, age, gender):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.gender = gender

    def __str__(self):
        return f"<Person name={self.first_name} {self.last_name}, age={self.age}, gender={self.gender}>"


p = Person("John", "Doe", 99, "Male")

# save the object
with open("person.pickle", "wb") as file:
    pickle.dump(p, file)

# load the object
with open("person.pickle", "rb") as file:
    p2 = pickle.load(file)

print(p)
print(p2)