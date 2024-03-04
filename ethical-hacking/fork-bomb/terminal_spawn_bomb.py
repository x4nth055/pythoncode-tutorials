"""A terminal spawn bomb that infinitely opens a new terminal window on the host system.
Be careful when running this script, as it overwhelms the system with terminal windows.
The time.sleep() is introduced to test the script."""
import os
import subprocess
import time

# List of common terminal emulators
terminal_emulators = [
    "gnome-terminal",  # GNOME
    "konsole",         # KDE
    "xfce4-terminal",  # XFCE
    "lxterminal",      # LXDE
    "mate-terminal",   # MATE
    "terminator",
    "xterm",
    "urxvt"
]

def open_terminal():
    for emulator in terminal_emulators:
        try:
            if subprocess.call(["which", emulator], stdout=subprocess.DEVNULL) == 0:
                os.system(f"{emulator} &")
                return True
        except Exception as e:
            continue
    print("No known terminal emulator found!")
    return False

while True:
    if os.name == "nt":
        os.system("start cmd")
    else:
        if not open_terminal():
            break  # Break the loop if no terminal emulator is found
    # Introduce a sleep of 500 ms to intentionally slow down the loop so you can stop the script.
    time.sleep(0.5)  # Adjust sleep time as needed to make it slower.
