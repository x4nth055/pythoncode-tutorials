# importing everything from tkinter
from tkinter import *
# importing ttk widgets from tkinter
from tkinter import ttk
import requests
# tkinter message box for displaying errors
from tkinter.messagebox import showerror


API_KEY = 'put your API key here'

# the Standard request url
url = f'https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD'

# making the Standard request to the API
response = requests.get(f'{url}').json()

# converting the currencies to dictionaries
currencies = dict(response['conversion_rates'])

def convert_currency():
    # will execute the code when everything is ok
    try:
        # getting currency from first combobox
        source = from_currency_combo.get()
        # getting currency from second combobox
        destination = to_currency_combo.get()
        # getting amound from amount_entry
        amount = amount_entry.get()
        # sending a request to the Pair Conversion url and converting it to json
        result = requests.get(f'https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{source}/{destination}/{amount}').json()
        # getting the conversion result from response result
        converted_result = result['conversion_result']
        # formatting the results
        formatted_result = f'{amount} {source} = {converted_result} {destination}'
        # adding text to the empty result label
        result_label.config(text=formatted_result)
        # adding text to the empty time label
        time_label.config(text='Last updated,' + result['time_last_update_utc'])
    # will catch all the errors that might occur 
    # ConnectionTimeOut, JSONDecodeError etc
    except:
        showerror(title='Error', message="An error occurred!!. Fill all the required field or check your internet connection.")


# creating the main window
window = Tk()

# this gives the window the width(310), height(320) and the position(center)
window.geometry('310x340+500+200')

# this is the title for the window
window.title('Currency Converter')

# this will make the window not resizable, since height and width is FALSE
window.resizable(height=FALSE, width=FALSE)

# colors for the application
primary = '#081F4D'
secondary = '#0083FF'
white = '#FFFFFF'

# the top frame
top_frame = Frame(window, bg=primary, width=300, height=80)
top_frame.grid(row=0, column=0)

# label for the text Currency Converter
name_label = Label(top_frame, text='Currency Converter', bg=primary, fg=white, pady=30, padx=24, justify=CENTER, font=('Poppins 20 bold'))
name_label.grid(row=0, column=0)


# the bottom frame
bottom_frame = Frame(window, width=300, height=250)
bottom_frame.grid(row=1, column=0)

# widgets inside the bottom frame
from_currency_label = Label(bottom_frame, text='FROM:', font=('Poppins 10 bold'), justify=LEFT)
from_currency_label.place(x=5, y=10)

to_currency_label = Label(bottom_frame, text='TO:', font=('Poppins 10 bold'), justify=RIGHT)
to_currency_label.place(x=160, y=10)

# this is the combobox for holding from_currencies
from_currency_combo = ttk.Combobox(bottom_frame, values=list(currencies.keys()), width=14, font=('Poppins 10 bold'))
from_currency_combo.place(x=5, y=30)

# this is the combobox for holding to_currencies
to_currency_combo = ttk.Combobox(bottom_frame, values=list(currencies.keys()), width=14, font=('Poppins 10 bold'))
to_currency_combo.place(x=160, y=30)

# the label for AMOUNT
amount_label = Label(bottom_frame, text='AMOUNT:', font=('Poppins 10 bold'))
amount_label.place(x=5, y=55)

# entry for amount
amount_entry = Entry(bottom_frame, width=25, font=('Poppins 15 bold'))
amount_entry.place(x=5, y=80)

# an empty label for displaying the result
result_label = Label(bottom_frame, text='', font=('Poppins 10 bold'))
result_label.place(x=5, y=115)

# an empty label for displaying the time
time_label = Label(bottom_frame, text='', font=('Poppins 10 bold'))
time_label.place(x=5, y=135)

# the clickable button for converting the currency
convert_button = Button(bottom_frame, text="CONVERT", bg=secondary, fg=white, font=('Poppins 10 bold'), command=convert_currency)
convert_button.place(x=5, y=165)


# this runs the window infinitely until it is closed
window.mainloop()