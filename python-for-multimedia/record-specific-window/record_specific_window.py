import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys

# the window name, e.g "notepad", "Chrome", etc.
window_name = sys.argv[1]

# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# frames per second
fps = 12.0
# the time you want to record in seconds
record_seconds = 10
# search for the window, getting the first matched window with the title
w = gw.getWindowsWithTitle(window_name)[0]
# activate the window
w.activate()
# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, fps, tuple(w.size))

for i in range(int(record_seconds * fps)):
    # make a screenshot
    img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))
    # convert these pixels to a proper numpy array to work with OpenCV
    frame = np.array(img)
    # convert colors from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # write the frame
    out.write(frame)
    # show the frame
    cv2.imshow("screenshot", frame)
    # if the user clicks q, it exits
    if cv2.waitKey(1) == ord("q"):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()
out.release()

