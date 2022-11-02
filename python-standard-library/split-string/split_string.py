#Declare Two Variables
variable1 = "Splitting a string"
variable2 = 'Splitting another string'

#Splitting The Variables
print(variable1.split())
print(variable2.split())

#Splitting The Variables
print(variable1.split())
print(variable2.split(","))

#Declare Two Variables
variable1 = "Splitting*a*string"
variable2 = 'Splitting,another,string'
#Splitting The Variables
print(variable1.split("*"))
print(variable2.split(","))

#Splitting The Variables
print(variable1.split("*")[2])
print(variable2.split(",")[0])

#Declare The Variable
variable = "Splitting a string"
#Use The Maxsplit
print(variable.split(" ", maxsplit=1))

#Declare The Variable
variable = "Splitting a string"
#Split The String By Characters
print(list(variable))