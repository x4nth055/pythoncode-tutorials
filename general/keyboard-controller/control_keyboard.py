import keyboard

# registering a hotkey that replaces one typed text with another
# replaces every "@email" followed by a space with my actual email
keyboard.add_abbreviation("@email", "email@domain.com")

# invokes a callback everytime a hotkey is pressed
keyboard.add_hotkey("ctrl+alt+p", lambda: print("CTRL+ALT+P Pressed!"))

# check if a ctrl is pressed
print(keyboard.is_pressed('ctrl'))

# press space
keyboard.send("space")

# sends artificial keyboard events to the OS
# simulating the typing of a given text
# setting 0.1 seconds to wait between keypresses to look fancy
keyboard.write("Python Programming is always fun!", delay=0.1)

# record all keyboard clicks until esc is clicked
events = keyboard.record('esc')
# play these events
keyboard.play(events)

# remove all keyboard hooks in use
keyboard.unhook_all()

