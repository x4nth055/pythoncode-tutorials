from tkinter import *
from tkinter import ttk
from datetime import date
from tkinter.messagebox import showerror


# the function for calculating the age
def calculate_age():
    # the try/except block
    try:
        # getting current date
        today = date.today()
        # getting day from the day entry
        day = int(day_entry.get())
        # getting month from the month entry
        month = int(month_entry.get())
        # getting year from the year entry
        year = int(year_entry.get())
        # creating a date object
        birthdate = date(year, month, day)
        # calculating the age
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        # displaying the age using the age result label
        age_result.config(text='You are ' + str(age) + ' years old')
    # if an error occurs the showerror window will pop up
    except:
        showerror(title='Error', message='An error occurred while trying to ' \
                    'calculate age\nThe following could ' \
                    'be the causes:\n->Invalid input data\n->An empty field/fields\n'\
                     'Make sure you enter valid data and fill all the fields')



# creating the main window
window = Tk()
# the title for the window
window.title('Age Calculator')
# the dimensions and position of the windodw
window.geometry('500x260+430+300')
# making the window nonresizabale
window.resizable(height=FALSE, width=FALSE)

# the canvas to contain all the widgets
canvas = Canvas(window, width=500, height=400)
canvas.pack()

# ttk styles for the labels
label_style = ttk.Style()
label_style.configure('TLabel', foreground='#000000', font=('OCR A Extended', 14))

# ttk styles for the button
button_style = ttk.Style()
button_style.configure('TButton', foreground='#000000', font=('DotumChe', 16))

# ttk styles for the entries
entry_style = ttk.Style()
entry_style.configure('TEntry', font=('Dotum', 15))

# the label for displaying the big text
big_label = Label(window, text='AGE CALCULATOR', font=('OCR A Extended', 25))

# placing the big label inside the canvas
canvas.create_window(245, 40, window=big_label)


# label and entry for the day
day_label = ttk.Label(window, text='Day:', style='TLabel')
day_entry = ttk.Entry(window, width=15, style='TEntry')

# label and entry for the month
month_label = ttk.Label(window, text='Month:', style='TLabel')
month_entry = ttk.Entry(window, width=15, style='TEntry')

# label and entry for the year
year_label = ttk.Label(window, text='Year:', style='TLabel')
year_entry = ttk.Entry(window, width=15, style='TEntry')

# the button 
calculate_button = ttk.Button(window, text='Calculate Age', style='TButton', command=calculate_age)

# label for display the calculated age
age_result = ttk.Label(window, text='', style='TLabel')


# adding the day label and entry inside the canvas
canvas.create_window(114, 100, window=day_label)
canvas.create_window(130, 130, window=day_entry)

# adding the month label and entry inside the canvas
canvas.create_window(250, 100, window=month_label)
canvas.create_window(245, 130, window=month_entry)

# adding the year label and entry inside the canvas
canvas.create_window(350, 100, window=year_label)
canvas.create_window(360, 130, window=year_entry)

# adding the age_result and entry inside the canvas
canvas.create_window(245, 180, window=age_result)

# adding the calculate button inside the canvas
canvas.create_window(245, 220, window=calculate_button)


# runs the window infinitely until uses closes it
window.mainloop()