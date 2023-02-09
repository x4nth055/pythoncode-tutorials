# this imports everything from the tkinter module
from tkinter import *
# importing the ttk module from tkinter that's for styling widgets
from tkinter import ttk
# importing a Text field with the scrollbar
from tkinter.scrolledtext import ScrolledText
# imports the re module
import re
# this imports nltk
import nltk
# importing all the words from nltk
from nltk.corpus import words
# importing an askyesno message box from tkinter.message
from tkinter.messagebox import askyesno

# this will download the words
nltk.download('words')

# we are creating a SpellingChecker class
class SpellingChecker:
    # a special method, always called when an instance is created
    def __init__(self, master):
        # defining a style for the label
        style = ttk.Style()
        # configuring the style, TLabel is the style name
        style.configure('TLabel', foreground='#000000', font=('OCR A Extended', 25))
        # variable for tracking white space, default is 0
        self.old_spaces = 0
        # creating the main window
        self.master = master
        # giving title to the main window
        self.master.title('Real-Time Spelling Checker')
        # defining dimensions and position for the main window
        self.master.geometry('580x500+440+180')
        # adding an icon to the main window
        self.master.iconbitmap(self.master, 'spell-check.ico')
        # making the main window non-resizable
        self.master.resizable(height=FALSE, width=FALSE)
        # this is for closing the window via the close() function
        self.master.protocol('WM_DELETE_WINDOW', self.close)
        # creating the label to display the big text
        self.label = ttk.Label(self.master, text='Real-Time Spelling Checker', style='TLabel')
        # adding the label to the main window using grid geometry manager
        self.label.grid(row=0, column=0, columnspan=10, padx=5, pady=25)
        # creating a scollable Text field
        self.text = ScrolledText(self.master, font=("Helvetica", 15), width=50, height=15)
        # bing the scrollable Text field to an event
        self.text.bind('<KeyRelease>', self.check)
        # adding the scrollable Text field to the main window using grid geometry manager
        self.text.grid(row=1, column=0, padx=5, pady=5, columnspan=10)
        
    # the function for closing the application
    def close(self):
        # this will ask the user whether to close or not
        # if the value is yes/True the window will close
        if askyesno(title='Close Real-Time Spelling Checker', message='Are you sure you want to close the application?'):
            # this destroys the window
            self.master.destroy()
    
    # this is the function for checking spelling in real-time
    def check(self, event):
        # getting all the content from the ScrolledText via get() function
        # 1.0 is the starting point and END is the end point of the ScrolledText content 
        content = self.text.get('1.0', END)
        # getting all the white spaces from the content
        space_count = content.count(' ')
        # checking if the space_count is not equal to self.old_spaces
        if space_count != self.old_spaces:
            # updating the self.old_spaces to space_count
            self.old_spaces = space_count
            # this loops through all the tag names
            # and deletes them if the word is valid
            for tag in self.text.tag_names():
                self.text.tag_delete(tag)
            # splitting the content by white space
            # and looping through the split content to get a single word
            for word in content.split(' '):
                # with the sub() function we are removing special characters from the word
                # replacing the special character with nothing
                # the target is word.lower()
                # checking if the cleaned lower case word is not in words
                if re.sub(r'[^\w]', '', word.lower()) not in words.words():
                    # gets the position of the invalid word
                    position = content.find(word)
                    # adding a tag to the invalid word
                    self.text.tag_add(word, f'1.{position}', f'1.{position + len(word)}')
                    # changing the color of the invalid word to red
                    self.text.tag_config(word, foreground='red')
        
    
    
# creating the root winding using Tk() class
root = Tk()
# instantiating/creating object app for class SpellingChecker 
app = SpellingChecker(root)
# calling the mainloop to run the app infinitely until user closes it
root.mainloop()
