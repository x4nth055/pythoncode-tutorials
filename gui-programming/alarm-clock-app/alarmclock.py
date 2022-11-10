from tkinter import *
import datetime
import time
from playsound import playsound
from tkinter import messagebox
from threading import *


root = Tk()  # initializes tkinter to create display window
root.geometry('450x250')  # width and height of the window
root.resizable(0, 0)  # sets fix size of window
root.title(' Alarm Clock')  # gives the window a title


addTime = Label(root, fg="red", text="Hour     Min     Sec",
                font='arial 12 bold').place(x=210)
setYourAlarm = Label(root, text="Set Time(24hrs): ",
                     bg="grey", font="arial 11 bold").place(x=80, y=40)
hour = StringVar()
min = StringVar()
sec = StringVar()

# make the time input fields
hourTime = Entry(root, textvariable=hour, relief=RAISED, width=4, font=(20)).place(x=210, y=40)
minTime = Entry(root, textvariable=min, width=4, font=(20)).place(x=270, y=40)
secTime = Entry(root, textvariable=sec, width=4, font=(20)).place(x=330, y=40)


def start_alarm():
    t1 = Thread(target=alarm)
    t1.start()


def alarm():
    while True:
        set_alarm_time = f"{hour.get()}:{min.get()}:{sec.get()}"
        # sleep for 1s to update the time every second
        time.sleep(1)
        # Get current time
        actual_time = datetime.datetime.now().strftime("%H:%M:%S")
        FMT = '%H:%M:%S'
        # get time remaining
        time_remaining = datetime.datetime.strptime(
            set_alarm_time, FMT) - datetime.datetime.strptime(actual_time, FMT)
        # displays current time
        CurrentLabel = Label(
            root, text=f'Current time: {actual_time}', fg='black')
        CurrentLabel.place(relx=0.2, rely=0.8, anchor=CENTER)
        # displays alarm time
        AlarmLabel = Label(
            root, text=f'Alarm time: {set_alarm_time}', fg='black')
        AlarmLabel.place(relx=0.2, rely=0.9, anchor=CENTER)
        # displays time remaining
        RemainingLabel = Label(
            root, text=f'Remaining time: {time_remaining}', fg='red')
        RemainingLabel.place(relx=0.7, rely=0.8, anchor=CENTER)
        # Check whether set alarm is equal to current time
        if actual_time == set_alarm_time:
            # Playing sound
            playsound('audio.mp3')
            messagebox.showinfo("TIME'S UP!!!")


# create a button to set the alarm
submit = Button(root, text="Set Your Alarm", fg="red", width=20,
                command=start_alarm, font=("arial 20 bold")).pack(pady=80, padx=120)
# run the program
root.mainloop()
