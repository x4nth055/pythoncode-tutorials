# this imports everything from the tkinter module
from tkinter import *
# importing the ttk module from tkinter that's for styling widgets
from tkinter import ttk
# importing message boxes like showinfo, showerror, askyesno from tkinter.messagebox
from tkinter.messagebox import showinfo, showerror, askyesno
# importing filedialog from tkinter
from tkinter import filedialog as fd 
# this imports the qrcode module
import qrcode
# this imports the cv2 module
import cv2



# the function to close the window
def close_window():
    # this will ask the user whether to close or not
    # if the value is yes/True the window will close
    if askyesno(title='Close QR Code Generator-Detector', message='Are you sure you want to close the application?'):
        # this destroys the window
        window.destroy()
        
        
        
        
# the function for generating the QR Code
def generate_qrcode():
    # getting qrcode data from data_entry via get() function
    qrcode_data = str(data_entry.get())
    # getting the qrcode name from the filename_entry via get() function
    qrcode_name = str(filename_entry.get())
    # checking if the qrcode_name/filename_entry is empty
    if qrcode_name == '':
        # if its empty display an error message to the user
        showerror(title='Error', message='An error occurred' \
                   '\nThe following is ' \
                    'the cause:\n->Empty filename entry field\n' \
                    'Make sure the filename entry field is filled when generating the QRCode')

    # the else statement will execute when the qrcode_name/filename_entry is filled
    else:
        # confirm from the user whether to generate QR code or not
        if askyesno(title='Confirmation', message=f'Do you want to create a QRCode with the provided information?'):
            # the try block for generating the QR Code
            try:
                # Creating an instance of QRCode class
                qr = qrcode.QRCode(version = 1, box_size = 6, border = 4)
                # Adding data to the instance 'qr'
                qr.add_data(qrcode_data)
                # 
                qr.make(fit = True)
                # the name for the QRCode
                name = qrcode_name + '.png'
                # making the QR code
                qrcode_image = qr.make_image(fill_color = 'black', back_color = 'white')
                # saving the QR code
                qrcode_image.save(name)
                # making the Image variable global
                global Image
                # opening the qrcode image file
                Image = PhotoImage(file=f'{name}')
                # displaying the image on the canvas via the image label
                image_label1.config(image=Image)
                # the button for resetting or clearing the QR code image on the canvas
                reset_button.config(state=NORMAL, command=reset)
        
            # this will catch all the errors that might occur
            except:
                showerror(title='Error', message='Please provide a valid filename')
        
# the function for resetting or clearing the image label
def reset():
    # confirming if the user wants to reset or not
    if askyesno(title='Reset', message='Are you sure you want to reset?'):
        # if yes reset the label
        image_label1.config(image='')
        # and disable the button again
        reset_button.config(state=DISABLED)
        
        
# the function to open file dialogs  
def open_dialog():
    # getting the file name via the askopenfilename() function
    name = fd.askopenfilename()
    # deleting every data from the file_entry
    file_entry.delete(0, END)
    # inserting the file in the file_entry
    file_entry.insert(0, name)


# the function to detect the QR codes
def detect_qrcode():
    
    # getting the image file from the file entry via get() function
    image_file = file_entry.get()
    # checking if the image_file is empty
    if image_file == '':
        # show error when the image_file entry is empty
        showerror(title='Error', message='Please provide a QR Code image file to detect')
    
    # executes when the image_file is not empty
    else:
        # code inside the try will detect the QR codes
        try:
            # reading the image file with cv2 
            qr_img = cv2.imread(f'{image_file}')  
            # using the QRCodeDetector() function  
            qr_detector = cv2.QRCodeDetector()  
            # making the qrcodde_image global
            global qrcode_image
            # opening the qrcode_image using the PhotoImage
            qrcode_image = PhotoImage(file=f'{image_file}')
            # displaying the image via the image label
            image_label2.config(image=qrcode_image)
            # using the detectAndDecode() function detect and decode the QR code
            data, pts, st_code = qr_detector.detectAndDecode(qr_img)  
            # displaying data on the data_label
            data_label.config(text=data)
            
        # this catches any errors that might occur
        except:
            # displaying an error message
            showerror(title='Error', message='An error occurred while detecting data from the provided file' \
                   '\nThe following could be ' \
                    'the cause:\n->Wrong image file\n' \
                    'Make sure the image file is a valid QRCode')





# creating the window using the Tk() class
window = Tk()
# creates title for the window
window.title('QR Code Generator-Detector')
# adding the window's icon
window.iconbitmap(window, 'icon.ico')
# dimensions and position of the window
window.geometry('500x480+440+180')
# makes the window non-resizable
window.resizable(height=FALSE, width=FALSE)
# this is for closing the window via the close_window() function
window.protocol('WM_DELETE_WINDOW', close_window)



"""Styles for the widgets, labels, entries, and buttons"""
# style for the labels
label_style = ttk.Style()
label_style.configure('TLabel', foreground='#000000', font=('OCR A Extended', 11))

# style for the entries
entry_style = ttk.Style()
entry_style.configure('TEntry', font=('Dotum', 15))

# style for the buttons
button_style = ttk.Style()
button_style.configure('TButton', foreground='#000000', font=('DotumChe', 10))

# creating the Notebook widget
tab_control = ttk.Notebook(window)

# creating the two tabs with the ttk.Frame()
first_tab = ttk.Frame(tab_control)
second_tab = ttk.Frame(tab_control)

# adding the two tabs to the Notebook
tab_control.add(first_tab, text='QR Code Generator')
tab_control.add(second_tab, text='QR Code Detector')
# this makes the Notebook fill the entire main window so that its visible
tab_control.pack(expand=1, fill="both")


# creates the canvas for containing all the widgets in the first tab
first_canvas = Canvas(first_tab, width=500, height=480)
# packing the canvas to the first tab
first_canvas.pack()

# creates the canvas for containing all the widgets in the second tab
second_canvas = Canvas(second_tab, width=500, height=480)
# packing the canvas to the second tab
second_canvas.pack()


"""Widgets for the first tab"""

# creating an empty label
image_label1 = Label(window)
# adding the label to the canvas
first_canvas.create_window(250, 150, window=image_label1)

# creating a ttk label
qrdata_label = ttk.Label(window, text='QRcode Data', style='TLabel')
# creating a ttk entry
data_entry = ttk.Entry(window, width=55, style='TEntry')

# adding the label to the canvas
first_canvas.create_window(70, 330, window=qrdata_label)
# adding the entry to the canvas
first_canvas.create_window(300, 330, window=data_entry)

# creating a ttk label
filename_label = ttk.Label(window, text='Filename', style='TLabel')
# creating a ttk entry
filename_entry = ttk.Entry(width=55, style='TEntry')

# adding the label to the canvas
first_canvas.create_window(84, 360, window=filename_label)
# adding the entry to the canvas
first_canvas.create_window(300, 360, window=filename_entry)


# creating the reset button in a disabled mode
reset_button = ttk.Button(window, text='Reset', style='TButton', state=DISABLED)
# creating the generate button
generate_button = ttk.Button(window, text='Generate QRCode', style='TButton',  command=generate_qrcode)

# adding the reset button to the canvas
first_canvas.create_window(300, 390, window=reset_button)
# adding the generate button to the canvas
first_canvas.create_window(410, 390, window=generate_button)


"""Below are the widgets for the second tab"""

# creating the second image label
image_label2 = Label(window)
# creating the data label
data_label = ttk.Label(window)

# adding the second image label to the second_canvas
second_canvas.create_window(250, 150, window=image_label2)
# adding the data label to the canvas
second_canvas.create_window(250, 300, window=data_label)

# creating the file_entry
file_entry = ttk.Entry(window, width=60, style='TEntry')
# creating the browse button
browse_button = ttk.Button(window, text='Browse', style='TButton', command=open_dialog)

# adding the entry to the canvas
second_canvas.create_window(200, 350, window=file_entry)
# adding the generate button to the canvas
second_canvas.create_window(430, 350, window=browse_button)

# creating the detect button
detect_button = ttk.Button(window, text='Detect QRCode', style='TButton', command=detect_qrcode)
# adding the detect button to the canvas
second_canvas.create_window(65, 385, window=detect_button)


# run the main window infinitely
window.mainloop()