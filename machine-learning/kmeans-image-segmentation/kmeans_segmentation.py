import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

print(pixel_values.shape)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(labels)
print(centers.shape)

# convert back to 8 bit values
centers = np.uint8(centers)

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)

# show the image
plt.imshow(segmented_image)
plt.show()

# disable only the cluster number 2
masked_image = np.copy(image)
masked_image[labels == 2] = [0, 0, 0]

# show the image
plt.imshow(masked_image)
plt.show()