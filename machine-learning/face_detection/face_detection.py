import cv2


# loading the test image
image = cv2.imread("kids.jpg")

# converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# detect all the faces in the image
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
# print the number of faces detected
print(f"{len(faces)} faces detected in the image.")

# for every face, draw a blue rectangle
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# save the image with rectangles
cv2.imwrite("kids_detected.jpg", image)


