import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(grayscale, 30, 100)

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), color=(20, 220, 20), thickness=3)

# show the image
plt.imshow(image)
plt.show()