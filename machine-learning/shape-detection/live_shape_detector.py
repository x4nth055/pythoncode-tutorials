import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    # convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform edge detection
    edges = cv2.Canny(grayscale, 30, 100)
    # detect lines in the image using hough lines technique
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
    # iterate over the output lines and draw them
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # show images
    cv2.imshow("image", image)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()