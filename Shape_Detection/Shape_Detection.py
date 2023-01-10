import cv2
import numpy as np


path = "/Users/aminarshadi/Desktop/Computer-Vision/Shape_Detection/img.jpeg" # path of the image to read

image = cv2.imread(path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_blur = cv2.GaussianBlur(image_gray, (35,35), 1)

image_canny = cv2.Canny(image_blur, 60, 60)


def getContours(selected_image):

	contours, hierarchy = cv2.findContours(selected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	for ch in contours:

		area = cv2.contourArea(ch)

		if area > 142:

			cv2.drawContours(image, ch, -1, (255,0,0), 2)

			arc_length = cv2.arcLength(ch, True)

			approx = cv2.approxPolyDP(ch, 0.02 * arc_length, True)

			corner_number = len(approx)

			print("Corner number: " + str(corner_number))

			x, y, w, h = cv2.boundingRect(approx)

			cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

			if corner_number == 3:
				text = "Triangle"

			elif corner_number == 4:
				text = "Rectangle"

			elif corner_number == 5:
				text = "5"

			elif corner_number == 6:
				text = "6"

			elif corner_number == 8:
				text = "Circle"

			elif corner_number == 12:
				text = "10"

			else:
				text = "None"

			cv2.putText( image, text, (x + (w // 2) - 60, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

getContours(image_canny)


cv2.imshow("Image Canny", image_canny)

cv2.imshow("Image", image)

cv2.waitKey(0)