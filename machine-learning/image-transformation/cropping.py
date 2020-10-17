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

# get 200 pixels from 100 to 300 on both x-axis & y-axis
# change that if you will, just make sure you don't exceed cols & rows
cropped_img = img[100:300, 100:300]
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.imshow(cropped_img)
plt.show()
# save the resulting image to disk
plt.imsave("city_cropped.jpg", cropped_img)