# importing everything from tkinter
from tkinter import *

# the requests will be used for making requests to the API
import requests

# tkinter message box to display errors
from tkinter.messagebox import showerror


def predict_gender():
    # executes when code has no errors
    try:
        # getting the input from entry
        entered_name = name_entry.get()
        # making a request to the API, the user's entered name is injected in the url
        response = requests.get(f'https://api.genderize.io/?name={entered_name}').json()
        # getting name from the response
        name = response['name']
        # getting gender from the response  
        gender = response['gender']
        # getting probability from the response 
        probability = 100 * response['probability']
        # adding name to the label that was empty, the name is being uppercased
        name_label.config(text='The name is ' + name.upper())
        # adding gender to the label that was empty, the gender is being uppercased  
        gender_label.config(text='The gender is ' + gender.upper())
        # adding probability to the label that was empty
        probability_label.config(text='Am ' + str(probability) + '%' + ' accurate')
    # executes when errors are caught
    # KeyError, ConnectionTimeoutError   
    except:
        showerror(title='error', message='An error occurred!! Make sure you have internet connection or you have entered the correct data')


# colors for the application
gold = '#dca714'
brown = '#31251d'

# creating the main window
window = Tk()
# defining the demensions of the window, width(325), height(300), 500+200 center the window
window.geometry('325x300+500+200')
# this is the title of the application
window.title('Gender Predictor')
# this makes the window unresizable
window.resizable(height=FALSE, width=FALSE)

"""The two frames"""
# this is the top frame inside the main window
top_frame = Frame(window, bg=brown, width=325, height=80)
top_frame.grid(row=0, column=0)

# this is the bottom frame inside the main window
bottom_frame = Frame(window, width=300, height=250)
bottom_frame.grid(row=1, column=0)

# the label for the big title inside the top_frame
first_label = Label(top_frame, text='GENDER PREDICTOR', bg=brown, fg=gold, pady=10, padx=20, justify=CENTER, font=('Poppins 20 bold'))
first_label.grid(row=0, column=0)

# the label for the small text inside the top_frame
second_label = Label(top_frame, text='Give me any name and i will predict its gender', bg=brown, fg=gold, font=('Poppins 10'))
second_label.grid(row=1, column=0)

"""below are widgets inside the top_frame"""
# the name label
label = Label(bottom_frame, text='NAME:', font=('Poppins 10 bold'), justify=LEFT)
label.place(x=4, y=10)

# the entry for entering the user's name
name_entry = Entry(bottom_frame, width=25, font=('Poppins 15 bold'))
name_entry.place(x=5, y=35)

# the empty name label, it will be used to display the name
name_label = Label(bottom_frame, text='', font=('Poppins 10 bold'))
name_label.place(x=5, y=70)

# the empty gender label, it will be used to display the gender
gender_label = Label(bottom_frame, text='', font=('Poppins 10 bold'))
gender_label.place(x=5, y=90)

# the empty probability label, it will be used to display the gender probalility
probability_label = Label(bottom_frame, text='', font=('Poppins 10 bold'))
probability_label.place(x=5, y=110)

# the predict button
predict_button = Button(bottom_frame, text="PREDICT", bg=gold, fg=brown, font=('Poppins 10 bold'), command=predict_gender)
predict_button.place(x=5, y=140)

window.mainloop()