# this imports everything from the tkinter module
from tkinter import *
# importing the ttk module from tkinter that's for styling widgets
from tkinter import ttk
# importing message boxes like showinfo, showerror, askyesno from tkinter.messagebox
from tkinter.messagebox import showerror, askyesno
# importing the PyDictionary library
from PyDictionary import PyDictionary
# this package converts text to speech
import pyttsx3


# the function for closing the application
def close_window():
    # this will ask the user whether to close or not
    # if the value is yes/True the window will close
    if askyesno(title='Close Audio Dictionary', message='Are you sure you want to close the application?'):
        # this destroys the window
        window.destroy()


# function for searching the word meaning
def search_word():
    # getting the word from the entry using the get()
    word = word_entry.get()
    # checking if the word variable is empty
    if word == '':
        # message box to display if the word variable is empty
        showerror(title='Error', message='Please enter the word you wanna search for!!')
    # the else statement will execute if the word variable is not empty  
    else:
        # this will execute the code that will find the word meanings
        try:
            # creating a dictionary object
            dictionary = PyDictionary()
            # passing a word to the dictionary object
            meanings = dictionary.meaning(word)
            # deleting content in the text field
            text_field.delete(1.0, END)
            # inserting content(meanings) in the text field
            text_field.insert('1.0', meanings)
            # adding the word to the empty label
            word_label.config(text=word)
            # enabling the audio button to normal state
            audio_button.config(state=NORMAL, command=speak)
        # this will catch all the exceptions, No/slow internet connection, word with wrong spellings
        except:
            # display the error to the user
            showerror(title='Error', message='An error occurred while trying to search word meaning' \
                   '\nThe following could be ' \
                    'the cause:\n->No/Slow internet connection\n' \
                    '->Wrong word spelling\n' \
                    'Please make sure you have a stable internet connection&\nthe word spelling is correct')




# function to turn textual data into audio data
def speak():
    # getting the word from the entry
    word = word_entry.get()
    # initializing the pyttsx3 object
    engine = pyttsx3.init()
    # gets the speaking rate
    rate = engine.getProperty('rate')
    # setting the speaking rate
    engine.setProperty('rate', 125)
    # getting the available voices
    voices = engine.getProperty('voices')
    # seeting the second voice, the female voice
    engine.setProperty('voice', voices[1].id)
    # this function takes the word to be spoken
    engine.say(word)
    # this fucntion processes the voice commands
    engine.runAndWait()



# creates the window using Tk() fucntion
window = Tk()
# creates title for the window
window.title('Audio-Dictionary')
# this is for closing the window via the close_window() function
window.protocol('WM_DELETE_WINDOW', close_window)
# adding the window's icon
window.iconbitmap(window, 'dictionary.ico')
# dimensions and position of the window
window.geometry('560x480+430+180')
# makes the window non-resizable
window.resizable(height=FALSE, width=FALSE)

"""Styles for the widgets"""
# style for the big text label 
big_label_style = ttk.Style()
big_label_style.configure('big_label_style.TLabel', foreground='#000000', font=('OCR A Extended', 40))
# style for small text labels
small_label_style = ttk.Style()
small_label_style.configure('small_label_style.TLabel', foreground='#000000', font=('OCR A Extended', 15))
# style for the entry
entry_style = ttk.Style()
entry_style.configure('TEntry', font=('Dotum', 20))
# style for the two buttons
button_style = ttk.Style()
button_style.configure('TButton', foreground='#000000', font='DotumChe')

# creates the canvas for containing all the widgets
canvas = Canvas(window, width=480, height=560)
# packing the canvas
canvas.pack()
# creating a ttk label
text_label = ttk.Label(window, text='Audio Dictionary', style='big_label_style.TLabel')
# adding the label to the canvas
canvas.create_window(250, 55, window=text_label)
# creating a ttk entry
word_entry = ttk.Entry(window, width=73, style='TEntry')
# adding the entry to the canvas
canvas.create_window(230, 110, window=word_entry, height=35)
# loading the icon
search_icon = PhotoImage(file='search.png')
# creates dimensions of the icon
logo = search_icon.subsample(20, 20)
# creating a ttk button with a search icon
search_button = ttk.Button(window, image=logo, style='TButton', command=search_word)
# adding the entry to the canvas
canvas.create_window(468, 110, window=search_button)
# loading the icon
audio_icon = PhotoImage(file='speaker.png')
# creates dimensions of the logo
icon = audio_icon.subsample(10, 10)
word_label = ttk.Label(window, style='small_label_style.TLabel')
# adding the label to the canvas
canvas.create_window(80, 145, window=word_label)
# creating another ttk button with a speaker icon
audio_button = ttk.Button(window, image=icon, style='TButton', state=DISABLED)
# adding the entry to the canvas
canvas.create_window(25, 190, window=audio_button) 
# creating the text field
text_field = Text(window, height=15, width=60)
# adding the text field to the canvas
canvas.create_window(248, 340, window=text_field) 
# runs the window infinitely
window.mainloop()