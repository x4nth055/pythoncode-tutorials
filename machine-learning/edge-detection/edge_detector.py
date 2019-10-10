import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the grayscale image, if you want to show, uncomment 2 below lines
# plt.imshow(gray, cmap="gray")
# plt.show()

# perform the canny edge detector to detect image edges
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# show the detected edges
plt.imshow(edges, cmap="gray")
plt.show()