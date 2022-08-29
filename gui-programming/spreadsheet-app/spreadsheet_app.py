# Imports
from tkinter import *
import string
import sys
import ctypes

# Increase Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Define X and Y Axis Lists
xAxis = string.ascii_lowercase[0:7]
yAxis = range(0, 12)

# Cells will hold the strings vars and the lables 
cells = {}

# Open the content of the given file
# if one was provided and save it as a two 
# dimensional list.
CsvContent = ''
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        CsvContent = f.read().split('\n')
        for i, layer in enumerate(CsvContent):
            CsvContent[i] = layer.split(',')

# Make a new Top Level Element (Window)
root = Tk()

# Set the the title to also mention the given file name
# if there is one
title = "Spreadsheet App" if CsvContent == '' else f"Spreadsheet App - {sys.argv[1]}"
root.title(title)

# Evaluating a cell
def evaluateCell(cellId, *i):

    # Get the content from the string var
    # and make it lowercase
    content = cells[cellId][0].get()
    content = content.lower()

    # get the reference to the label
    label = cells[cellId][1]

    # if the cell starts with a = it is evaluated
    if content.startswith('='):

        # Loop through all cells ...
        for cell in cells:

            # ... and see if their name appears in this cell
            if cell in content.lower():

                # if it is then replace the name occurences 
                # with the evaluated content from there.
                content = content.replace(cell, str(evaluateCell(cell)))

        # Get the content without the = and try to evaluate it
        content = content[1:]
        try:
            content = eval(content)
        except:
            content = 'NAN'
        label['text'] = content
        return content

    # If not, the label just shows the content
    else:
        label['text'] = content
        return content

# Call the eval function for every cell every ten milliseconds.
def updateAllCells():

    # Call it again
    root.after(10, updateAllCells)

    # Loop through all cells
    for cell in cells:
        evaluateCell(cell)


# Display the Y axis lables
for y in yAxis:
    label = Label(root, text = y, width=5, background='white')
    label.grid(row=y + 1, column=0)

# Display the X axis lables with enumerate
for i, x in enumerate(xAxis):
    label = Label(root, text = x, width=35, background='white')
    label.grid(row=0, column=i + 1, sticky='n')


# Display the Cells, by using a nested loop
for y in yAxis:
    for xcoor, x in enumerate(xAxis):

        # Generate a Unique ID for the cell with the coordinates
        id = f'{x}{y}'

        # Make String Var associated with the Cell
        var = StringVar(root, '', id)

        # Make Entry and label, offset each axis by one because of the lables
        e = Entry(root, textvariable=var, width=30)
        e.grid(row=y + 1, column=xcoor + 1)

        label = Label(root, text = '', width=5)
        label.grid(row=y + 1, column=xcoor + 1, sticky='e')

        # Save the string var and a reference to the lables in the dictionary
        cells[id] = [var, label]

        # Insert CSV content if it possible
        if CsvContent != '':
            try:
                var.set(CsvContent[y][xcoor])
            except:
                pass

# Start the updating Process
updateAllCells()

# Run the Mainloop
root.mainloop()
