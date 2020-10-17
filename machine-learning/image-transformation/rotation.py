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

#angle from degree to radian
angle = np.radians(10)
#transformation matrix for Rotation
M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
# apply a perspective transformation to the image
rotated_img = cv2.warpPerspective(img, M, (int(cols),int(rows)))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.imshow(rotated_img)
plt.show()
# save the resulting image to disk
plt.imsave("city_rotated.jpg", rotated_img)