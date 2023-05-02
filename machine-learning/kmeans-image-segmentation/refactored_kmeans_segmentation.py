import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_image(file_path):
    """Read the image and convert it to RGB."""
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    """Reshape the image to a 2D array of pixels and 3 color values (RGB) and convert to float."""
    pixel_values = image.reshape((-1, 3))
    return np.float32(pixel_values)

def perform_kmeans_clustering(pixel_values, k=3):
    """Perform k-means clustering on the pixel values."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, labels, np.uint8(centers)

def create_segmented_image(pixel_values, labels, centers):
    """Create a segmented image using the cluster centroids."""
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

def create_masked_image(image, labels, cluster_to_disable):
    """Create a masked image by disabling a specific cluster."""
    masked_image = np.copy(image).reshape((-1, 3))
    masked_image[labels.flatten() == cluster_to_disable] = [0, 0, 0]
    return masked_image.reshape(image.shape)

def display_image(image):
    """Display the image using matplotlib."""
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    image_path = sys.argv[1]
    k = int(sys.argv[2])
    # read the image
    image = read_image(image_path)
    # preprocess the image
    pixel_values = preprocess_image(image)
    # compactness is the sum of squared distance from each point to their corresponding centers
    compactness, labels, centers = perform_kmeans_clustering(pixel_values, k)
    # create the segmented image
    segmented_image = create_segmented_image(pixel_values, labels, centers)
    # display the image
    display_image(segmented_image)
    # disable only the cluster number 2 (turn the pixel into black)
    cluster_to_disable = 2
    # create the masked image
    masked_image = create_masked_image(image, labels, cluster_to_disable)
    display_image(masked_image)
