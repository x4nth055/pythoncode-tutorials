import cv2
import os


factor = 5


if not os.path.isdir("new-images"):
    os.mkdir("new-images")

for file in os.listdir("images"):
    file = os.path.join("images", file)
    basename = os.path.basename(file)
    img = cv2.imread(file)
    old_size = img.shape
    new_size = (img.shape[1] // factor, img.shape[0] // factor)
    print("Old size:", old_size)
    print("New size:", new_size)
    img = cv2.resize(img, new_size)
    new_filename = os.path.join("new-images", basename)
    cv2.imwrite(new_filename, img)
    print("Wrote", new_filename)