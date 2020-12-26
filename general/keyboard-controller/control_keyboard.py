import keyboard

# registering a hotkey that replaces one typed text with another
# replaces every "@email" followed by a space with my actual email
keyboard.add_abbreviation("@email", "email@domain.com")

# invokes a callback everytime a hotkey is pressed
keyboard.add_hotkey("ctrl+alt+p", lambda: print("CTRL+ALT+P Pressed!"))

# check if a ctrl is pressed
print(keyboard.is_pressed('ctrl'))

# press and release space
keyboard.send("space")

# multi-key, windows+d as example shows the desktop in Windows machines
keyboard.send("windows+d")

# send ALT+F4 in the same time, and then send space, 
# (be carful, this will close any current open window)
keyboard.send("alt+F4, space")

# press CTRL button
keyboard.press("ctrl")
# release the CTRL button
keyboard.release("ctrl")

# sends artificial keyboard events to the OS
# simulating the typing of a given text
# setting 0.1 seconds to wait between keypresses to look fancy
keyboard.write("Python Programming is always fun!", delay=0.1)

# record all keyboard clicks until esc is clicked
events = keyboard.record('esc')
# play these events
keyboard.play(events)
# print all typed strings in the events
print(list(keyboard.get_typed_strings(events)))

# log all pressed keys
keyboard.on_release(lambda e: print(e.name))

# remove all keyboard hooks in use
keyboard.unhook_all()

