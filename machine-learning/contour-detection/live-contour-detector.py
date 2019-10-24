import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 255 // 2, 255, cv2.THRESH_BINARY_INV)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours
    image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # show the images
    cv2.imshow("gray", gray)
    cv2.imshow("image", image)
    cv2.imshow("binary", binary)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
