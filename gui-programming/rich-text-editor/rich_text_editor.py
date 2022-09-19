from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import ctypes
from functools import partial
from json import loads, dumps

ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Setup
root = Tk()
root.geometry('600x600')

# Used to make title of the application
applicationName = 'Rich Text Editor'
root.title(applicationName)

# Current File Path
filePath = None

# initial directory to be the current directory
initialdir = '.'

# Define File Types that can be choosen
validFileTypes = (
    ("Rich Text File","*.rte"),
    ("all files","*.*")
)

# Setting the font and Padding for the Text Area
fontName = 'Bahnschrift'
padding = 60

# Infos about the Document are stored here
document = None

# Default content of the File
defaultContent = {
    "content": "",
    "tags": {
        'bold': [(), ()]
    },
}

# Transform rgb to hex
def rgbToHex(rgb):
    return "#%02x%02x%02x" % rgb  

# Add Different Types of Tags that can be added to the document.
tagTypes = {
    # Font Settings
    'Bold': {'font': f'{fontName} 15 bold'},
    'Italic': {'font': f'{fontName} 15 italic'},
    'Code': {'font': 'Consolas 15', 'background': rgbToHex((200, 200, 200))},

    # Sizes
    'Normal Size': {'font': f'{fontName} 15'},
    'Larger Size': {'font': f'{fontName} 25'},
    'Largest Size': {'font': f'{fontName} 35'},

    # Background Colors
    'Highlight': {'background': rgbToHex((255, 255, 0))},
    'Highlight Red': {'background': rgbToHex((255, 0, 0))},
    'Highlight Green': {'background': rgbToHex((0, 255, 0))},
    'Highlight Black': {'background': rgbToHex((0, 0, 0))},

    # Foreground /  Text Colors
    'Text White': {'foreground': rgbToHex((255, 255, 255))},
    'Text Grey': {'foreground': rgbToHex((200, 200, 200))},
    'Text Blue': {'foreground': rgbToHex((0, 0, 255))},
    'Text green': {'foreground': rgbToHex((0, 255, 0))},
    'Text Red': {'foreground': rgbToHex((255, 0, 0))},
}

# Handle File Events
def fileManager(event=None, action=None):
    global document, filePath

    # Open
    if action == 'open':
        # ask the user for a filename with the native file explorer.
        filePath = askopenfilename(filetypes=validFileTypes, initialdir=initialdir)


        with open(filePath, 'r') as f:
            document = loads(f.read())

        # Delete Content
        textArea.delete('1.0', END)
        
        # Set Content
        textArea.insert('1.0', document['content'])

        # Set Title
        root.title(f'{applicationName} - {filePath}')

        # Reset all tags
        resetTags()

        # Add To the Document
        for tagName in document['tags']:
            for tagStart, tagEnd in document['tags'][tagName]:
                textArea.tag_add(tagName, tagStart, tagEnd)
                print(tagName, tagStart, tagEnd)

    elif action == 'save':
        document = defaultContent
        document['content'] = textArea.get('1.0', END)

        for tagName in textArea.tag_names():
            if tagName == 'sel': continue

            document['tags'][tagName] = []

            ranges = textArea.tag_ranges(tagName)

            for i, tagRange in enumerate(ranges[::2]):
                document['tags'][tagName].append([str(tagRange), str(ranges[i+1])])

        if not filePath:
            # ask the user for a filename with the native file explorer.
            newfilePath = asksaveasfilename(filetypes=validFileTypes, initialdir=initialdir)
    
            # Return in case the User Leaves the Window without
            # choosing a file to save
            if newfilePath is None: return

            filePath = newfilePath

        if not filePath.endswith('.rte'):
            filePath += '.rte'

        with open(filePath, 'w') as f:
            print('Saving at: ', filePath)  
            f.write(dumps(document))

        root.title(f'{applicationName} - {filePath}')


def resetTags():
    for tag in textArea.tag_names():
        textArea.tag_remove(tag, "1.0", "end")

    for tagType in tagTypes:
        textArea.tag_configure(tagType.lower(), tagTypes[tagType])


def keyDown(event=None):
    root.title(f'{applicationName} - *{filePath}')


def tagToggle(tagName):
    start, end = "sel.first", "sel.last"

    if tagName in textArea.tag_names('sel.first'):
        textArea.tag_remove(tagName, start, end)
    else:
        textArea.tag_add(tagName, start, end)


textArea = Text(root, font=f'{fontName} 15', relief=FLAT)
textArea.pack(fill=BOTH, expand=TRUE, padx=padding, pady=padding)
textArea.bind("<Key>", keyDown)

resetTags()


menu = Menu(root)
root.config(menu=menu)

fileMenu = Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=fileMenu)

fileMenu.add_command(label="Open", command=partial(fileManager, action='open'), accelerator='Ctrl+O')
root.bind_all('<Control-o>', partial(fileManager, action='open'))

fileMenu.add_command(label="Save", command=partial(fileManager, action='save'), accelerator='Ctrl+S')
root.bind_all('<Control-s>', partial(fileManager, action='save'))

fileMenu.add_command(label="Exit", command=root.quit)


formatMenu = Menu(menu, tearoff=0)
menu.add_cascade(label="Format", menu=formatMenu)

for tagType in tagTypes:
    formatMenu.add_command(label=tagType, command=partial(tagToggle, tagName=tagType.lower()))


root.mainloop()
