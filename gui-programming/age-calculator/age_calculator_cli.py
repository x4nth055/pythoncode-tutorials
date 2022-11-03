from datetime import date

def calculate_age(day, month, year):
    # we are getting the current date using the today()
    today = date.today()
    # convering year, month and day into birthdate
    birthdate = date(year, month, day)
    # calculating the age 
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    # return the age value
    return age


# the try/except block
# the try will execute if there are no exceptions
try:
    # we are getting day, month and year using input() function
    day = input('Enter day:')
    month = input('Enter month:')
    year = input('Enter year:')
    # creating a variable called calculated_age and we are also calling the claculate_age function
    age_result = calculate_age(int(day), int(month), int(year))
    print(f'You are {age_result} years old')
    
# the except will catch all errors
except:
    print(f'Failed to calculate age, either day or month or year is invalid')
