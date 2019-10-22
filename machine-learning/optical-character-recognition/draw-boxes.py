import pytesseract
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image

# read the image using OpenCV
image = cv2.imread(sys.argv[1])

# make a copy of this image to draw in
image_copy = image.copy()

# the target word to search for
target_word = sys.argv[2]

# get all data from the image
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# get all occurences of the that word
word_occurences = [ i for i, word in enumerate(data["text"]) if word.lower() == target_word ]

for occ in word_occurences:
    # extract the width, height, top and left position for that detected word
    w = data["width"][occ]
    h = data["height"][occ]
    l = data["left"][occ]
    t = data["top"][occ]
    # define all the surrounding box points
    p1 = (l, t)
    p2 = (l + w, t)
    p3 = (l + w, t + h)
    p4 = (l, t + h)
    # draw the 4 lines (rectangular)
    image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=2)

plt.imsave("all_dog_words.png", image_copy)
plt.imshow(image_copy)
plt.show()


