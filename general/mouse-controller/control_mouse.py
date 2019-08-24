import mouse

# left click
mouse.click('left')

# right click
mouse.click('right')

# middle click
mouse.click('middle')

# get the position of mouse
print(mouse.get_position())
# In [12]: mouse.get_position()
# Out[12]: (714, 488)

# presses but doesn't release
mouse.hold('left')
# mouse.press('left')

# drag from (0, 0) to (100, 100) relatively with a duration of 0.1s
mouse.drag(0, 0, 100, 100, absolute=False, duration=0.1)

# whether a button is clicked
print(mouse.is_pressed('right'))

# move 100 right & 100 down
mouse.move(100, 100, absolute=False, duration=0.2)

# make a listener when left button is clicked
mouse.on_click(lambda: print("Left Button clicked."))
# make a listener when right button is clicked
mouse.on_right_click(lambda: print("Right Button clicked."))

# remove the listeners when you want
mouse.unhook_all()

# scroll down
mouse.wheel(-1)

# scroll up
mouse.wheel(1)

# record until you click right
events = mouse.record()

# replay these events
mouse.play(events[:-1])


