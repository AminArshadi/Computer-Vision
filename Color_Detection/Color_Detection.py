import cv2
import numpy as np


def empty(a):
	pass


path = "/Users/aminarshadi/Desktop/Computer-Vision/Color_Detection/img.jpeg" # path of the image to read

cv2.namedWindow("Adjuster")

cv2.resizeWindow("Adjuster", 700, 300)

cv2.createTrackbar("Hue min", "Adjuster", 100, 179, empty)

cv2.createTrackbar("Hue max", "Adjuster", 179, 179, empty)

cv2.createTrackbar("Sat min", "Adjuster", 41, 255, empty)

cv2.createTrackbar("Sat max", "Adjuster", 200, 255, empty)

cv2.createTrackbar("Value min", "Adjuster", 0, 255, empty)

cv2.createTrackbar("Value max", "Adjuster", 16, 255, empty)

while True:

	image = cv2.imread(path)

	image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	h_min = cv2.getTrackbarPos("Hue min", "Adjuster")

	h_max = cv2.getTrackbarPos("Hue max", "Adjuster")

	s_min = cv2.getTrackbarPos("Sat min", "Adjuster")

	s_max = cv2.getTrackbarPos("Sat max", "Adjuster")

	v_min = cv2.getTrackbarPos("Value min", "Adjuster")

	v_max = cv2.getTrackbarPos("Value max", "Adjuster")


	print(h_min, h_max, s_min, s_max, v_min, v_max)


	lower = np.array([h_min, s_min, v_min])

	upper = np.array([h_max, s_max, v_max])

	mask = cv2.inRange(image_HSV, lower, upper)

	new_image = cv2.bitwise_and(image, image, mask=mask)


	cv2.imshow("Image", image)

	cv2.imshow("Image HSV", image_HSV)

	cv2.imshow("Mask", mask)

	cv2.imshow("New Image", new_image)

	cv2.waitKey(1)