from tkinter import *
import os
import ctypes
import pathlib

# Increase Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

root = Tk()
# set a title for our file explorer main window
root.title('Simple Explorer')

root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=1)

def pathChange(*event):
    # Get all Files and Folders from the given Directory
    directory = os.listdir(currentPath.get())
    # Clearing the list
    list.delete(0, END)
    # Inserting the files and directories into the list
    for file in directory:
        list.insert(0, file)

def changePathByClick(event=None):
    # Get clicked item.
    picked = list.get(list.curselection()[0])
    # get the complete path by joining the current path with the picked item
    path = os.path.join(currentPath.get(), picked)
    # Check if item is file, then open it
    if os.path.isfile(path):
        print('Opening: '+path)
        os.startfile(path)
    # Set new path, will trigger pathChange function.
    else:
        currentPath.set(path)

def goBack(event=None):
    # get the new path
    newPath = pathlib.Path(currentPath.get()).parent
    # set it to currentPath
    currentPath.set(newPath)
    # simple message
    print('Going Back')


def open_popup():
    global top
    top = Toplevel(root)
    top.geometry("250x150")
    top.resizable(False, False)
    top.title("Child Window")
    top.columnconfigure(0, weight=1)
    Label(top, text='Enter File or Folder name').grid()
    Entry(top, textvariable=newFileName).grid(column=0, pady=10, sticky='NSEW')
    Button(top, text="Create", command=newFileOrFolder).grid(pady=10, sticky='NSEW')

def newFileOrFolder():
    # check if it is a file name or a folder
    if len(newFileName.get().split('.')) != 1:
        open(os.path.join(currentPath.get(), newFileName.get()), 'w').close()
    else:
        os.mkdir(os.path.join(currentPath.get(), newFileName.get()))
    # destroy the top
    top.destroy()
    pathChange()

top = ''

# String variables
newFileName = StringVar(root, "File.dot", 'new_name')
currentPath = StringVar(
    root,
    name='currentPath',
    value=pathlib.Path.cwd()
)
# Bind changes in this variable to the pathChange function
currentPath.trace('w', pathChange)

Button(root, text='Folder Up', command=goBack).grid(
    sticky='NSEW', column=0, row=0
)

# Keyboard shortcut for going up
root.bind("<Alt-Up>", goBack)

Entry(root, textvariable=currentPath).grid(
    sticky='NSEW', column=1, row=0, ipady=10, ipadx=10
)

# List of files and folder
list = Listbox(root)
list.grid(sticky='NSEW', column=1, row=1, ipady=10, ipadx=10)

# List Accelerators
list.bind('<Double-1>', changePathByClick)
list.bind('<Return>', changePathByClick)


# Menu
menubar = Menu(root)
# Adding a new File button
menubar.add_command(label="Add File or Folder", command=open_popup)
# Adding a quit button to the Menubar
menubar.add_command(label="Quit", command=root.quit)
# Make the menubar the Main Menu
root.config(menu=menubar)

# Call the function so the list displays
pathChange('')
# run the main program
root.mainloop()
