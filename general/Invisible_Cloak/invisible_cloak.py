#import libraries
import numpy as np 
import cv2

import time

cap = cv2.VideoCapture(0)

#time to adjust camera
time.sleep(5)

#background image display when cloak on myself
background = 0

#capturing the background
for i in range(50):

	ret, background = cap.read()

#code is running until webcam is not off
while(cap.isOpened()):
	
	ret, img = cap.read()

	if not ret:
		break

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#HSV values
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])

	#seprating the cloak part
	mask1 = cv2.inRange(hsv, lower_red, upper_red)

	lower_red = np.array([170, 70, 50])
	upper_red = np.array([180, 255, 255])
	mask2 = cv2.inRange(hsv, lower_red, upper_red)

	#OR
	mask1 = mask1 + mask2 

	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,
							np.ones((3,3), np.uint8), iterations=10) #noise removal
	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,
							np.ones((3,3), np.uint8), iterations=10)

	mask2 = cv2.bitwise_not(mask1)#except the cloak

	res1 = cv2.bitwise_and(background, background, mask=mask1)#used for segmentation of color
	res2 = cv2.bitwise_and(img, img, mask=mask2)#used to substitute the cloak part
	final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

	cv2.imshow('Welcome To Hogwarts!!',final_output)
	k= cv2.waitKey(10)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()


