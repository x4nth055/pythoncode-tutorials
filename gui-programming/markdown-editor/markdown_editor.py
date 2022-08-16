# Imports
from tkinter import *
import ctypes
import re

# Increas Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Setup
root = Tk()
root.title('Markdown Editor')
root.geometry('1000x600')

# Setting the Font globaly
root.option_add('*Font', 'Courier 15')


def changes(event=None):
    display['state'] = NORMAL

    # Clear the Display Area
    display.delete(1.0, END)

    # Insert the content of the Edit Area into the Display Area
    text = editor.get('1.0', END)

    # Save Raw Text for later
    textRaw = text

    # Remove Unwanted Characters
    text = ''.join(text.split('#'))
    text = ''.join(text.split('*'))

    display.insert(1.0, text)

    # Loop through each replacement, unpacking it fully
    for pattern, name, fontData, colorData, offset in replacements:

        # Get the location indices of the given pattern
        locations = search_re(pattern, textRaw, offset)

        print(f'{name} at {locations}')

        # Add tags where the search_re function found the pattern
        for start, end in locations:
            display.tag_add(name, start, end)

        # Configure the tag to use the specified font and color
        # to this every time to delete the previous tags
        display.tag_config(name, font=fontData, foreground=colorData)

    display['state'] = DISABLED

def search_re(pattern, text, offset):
    matches = []
    text = text.splitlines()
    for i, line in enumerate(text):
        for match in re.finditer(pattern, line):
            matches.append(
                (f"{i + 1}.{match.start()}", f"{i + 1}.{match.end() - offset}")
            )
    
    return matches

# Convert an RGB tuple to a HEX string using the % Operator
# 02 means print 2 characters
# x means hexadecimal
def rgbToHex(rgb):
    return "#%02x%02x%02x" % rgb

# Style Setup
editorBackground = rgbToHex((40, 40, 40))
editorTextColor = rgbToHex((230, 230, 230))
displayBackground = rgbToHex((60, 60, 60))
displayTextColor = rgbToHex((200, 200, 200))

caretColor = rgbToHex((255, 255, 255))

# Width of the Textareas in characters
width = 10

# Fonts
editorfontName = 'Courier'
displayFontName = 'Calibri'

# Font Sizes
normalSize = 15
h1Size = 40
h2Size = 30
h3Size = 20

# font Colors
h1Color = rgbToHex((240, 240, 240))
h2Color = rgbToHex((200, 200, 200))
h3Color = rgbToHex((160, 160, 160))

# Replacements tell us were to insert tags with the font and colors given
replacements = [
    [
        '^#[a-zA-Z\s\d\?\!\.]+$',
        'Header 1',
        f'{displayFontName} {h1Size}', 
        h1Color,
        0
    ], [
        '^##[a-zA-Z\s\d\?\!\.]+$',
        'Header 2', 
        f'{displayFontName} {h2Size}',
        h2Color,
        0
    ], [
        '^###[a-zA-Z\s\d\?\!\.]+$', 
        'Header 3', 
        f'{displayFontName} {h3Size}', 
        h3Color,
        0        
    ], [
        '\*.+?\*', 
        'Bold', 
        f'{displayFontName} {normalSize} bold', 
        displayTextColor,
        2
    ],
]

# Making the Editor Area
editor = Text(
    root,
    height=5,
    width=width,
    bg=editorBackground,
    fg=editorTextColor,
    border=30,
    relief=FLAT,
    insertbackground=caretColor
)
editor.pack(expand=1, fill=BOTH, side=LEFT)

# Bind <KeyReleas> so every change is registered
editor.bind('<KeyRelease>', changes)
editor.focus_set()

# Insert a starting text
editor.insert(INSERT, """#Heading 1

##Heading 2

###Heading 3
  
This is a *bold* move!


- Markdown Editor -

""")

# Making the Display Area
display = Text(
    root,
    height=5,
    width=width,
    bg=displayBackground,
    fg=displayTextColor,
    border=30,
    relief=FLAT,
    font=f"{displayFontName} {normalSize}",
)
display.pack(expand=1, fill=BOTH, side=LEFT)

# Disable the Display Area so the user cant write in it
# We will have to toggle it so we can insert text
display['state'] = DISABLED

# Starting the Application
changes()
root.mainloop()
