# Import
from tkinter import *
from tkinter import scrolledtext
from tkinter import filedialog
import ctypes
import sys
 
# Increas Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Setup Variables

appName = 'Simple Text Editor'
nofileOpenedString = 'New File'

currentFilePath = nofileOpenedString

# Viable File Types, when opening and saving files.
fileTypes = [("Text Files","*.txt"), ("Markdown","*.md")]

# Tkinter Setup
window = Tk()

# Set the first column to occupy 100% of the width
window.grid_columnconfigure(0, weight=1)

window.title(appName + " - " + currentFilePath)

# Window Dimensions in Pixel
window.geometry('500x400')

# Handler Functions
def fileDropDownHandeler(action):
    global currentFilePath

    # Opening a File
    if action == "open":
        file = filedialog.askopenfilename(filetypes = fileTypes)

        window.title(appName + " - " + file)

        currentFilePath = file

        with open(file, 'r') as f:
            txt.delete(1.0,END)
            txt.insert(INSERT,f.read())

    # Making a new File
    elif action == "new":
        currentFilePath = nofileOpenedString
        txt.delete(1.0,END)
        window.title(appName + " - " + currentFilePath)

    # Saving a file
    elif action == "save" or action == "saveAs":
        if currentFilePath == nofileOpenedString or action== 'saveAs':
            currentFilePath = filedialog.asksaveasfilename(filetypes = fileTypes)

        with open(currentFilePath, 'w') as f:
            f.write(txt.get('1.0','end'))

        window.title(appName + " - " + currentFilePath)

def textchange(event):
    window.title(appName + " - *" + currentFilePath)

# Widgets

# Text Area
txt = scrolledtext.ScrolledText(window, height=999)
txt.grid(row=1,sticky=N+S+E+W)

# Bind event in the widget to a function
txt.bind('<KeyPress>', textchange)

# Menu
menu = Menu(window)

# set tearoff to 0
fileDropdown = Menu(menu, tearoff=False)

# Add Commands and and their callbacks
fileDropdown.add_command(label='New', command=lambda: fileDropDownHandeler("new"))
fileDropdown.add_command(label='Open', command=lambda: fileDropDownHandeler("open"))

# Adding a seperator between button types.
fileDropdown.add_separator()
fileDropdown.add_command(label='Save', command=lambda: fileDropDownHandeler("save"))
fileDropdown.add_command(label='Save as', command=lambda: fileDropDownHandeler("saveAs"))

menu.add_cascade(label='File', menu=fileDropdown)

# Set Menu to be Main Menu
window.config(menu=menu)

# Enabling "open with" by looking if the second argument was passed.
if len(sys.argv) == 2:
    currentFilePath = sys.argv[1]

    window.title(appName + " - " + currentFilePath)

    with open(currentFilePath, 'r') as f:
        txt.delete(1.0,END)
        txt.insert(INSERT,f.read())

# Main Loop
window.mainloop()
