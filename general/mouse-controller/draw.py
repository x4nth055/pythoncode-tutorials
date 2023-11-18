import mouse
import math
import time

def draw_square(size):
    # click and hold the left mouse button
    mouse.press()
    mouse.move(size, 0, absolute=False, duration=0.2)
    mouse.move(0, size, absolute=False, duration=0.2)
    mouse.move(-size, 0, absolute=False, duration=0.2)
    mouse.move(0, -size, absolute=False, duration=0.2)
    # release the left mouse button
    mouse.release()


def draw_circle(radius):
    # click and hold the left mouse button
    mouse.press()
    # move the mouse in a circle
    for i in range(0, 360, 5):
        # convert degrees to radians
        angle = math.radians(i)
        # calculate the x and y coordinates
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # move the mouse to the calculated position
        mouse.move(x, y, absolute=False, duration=0.01)
    # release the left mouse button
    mouse.release()

if __name__ == "__main__":
    # Place the mouse at the starting point and then call
    draw_square(200)
    time.sleep(1)
    draw_circle(10)