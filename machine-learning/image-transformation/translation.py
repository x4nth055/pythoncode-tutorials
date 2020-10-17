import numpy as np
import cv2
import matplotlib.pyplot as plt

# read the input image
img = cv2.imread("city.jpg")
# convert from BGR to RGB so we can plot using matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# disable x & y axis
plt.axis('off')
# show the image
plt.imshow(img)
plt.show()

# get the image shape
rows, cols, dim = img.shape

# transformation matrix for translation
M = np.float32([[1, 0, 50],
                [0, 1, 50],
                [0, 0, 1]])
# apply a perspective transformation to the image
translated_img = cv2.warpPerspective(img, M, (cols, rows))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.imshow(translated_img)
plt.show()
# save the resulting image to disk
plt.imsave("city_translated.jpg", translated_img)