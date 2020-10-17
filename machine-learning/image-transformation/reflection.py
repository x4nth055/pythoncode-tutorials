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

# transformation matrix for x-axis reflection 
M = np.float32([[1,  0, 0   ],
                [0, -1, rows],
                [0,  0, 1   ]])
# transformation matrix for y-axis reflection
# M = np.float32([[-1, 0, cols],
#                 [ 0, 1, 0   ],
#                 [ 0, 0, 1   ]])
# apply a perspective transformation to the image
reflected_img = cv2.warpPerspective(img,M,(int(cols),int(rows)))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.imshow(reflected_img)
plt.show()
# save the resulting image to disk
plt.imsave("city_reflected.jpg", reflected_img)