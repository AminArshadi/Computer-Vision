import cv2
import numpy as np



def preProcessing(selected_image):

    image_gray = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)

    image_blur = cv2.GaussianBlur(image_gray, (5,5), 1)

    image_canny = cv2.Canny(image_blur, 200, 200)

    # make the edges thicker to imporve the edge recognition
    image_dilation = cv2.dilate(image_canny, np.ones((5,5)), iterations = 2)

    image_thresh = cv2.erode(image_dilation, np.ones((5,5)), iterations = 1)
    #
    return image_thresh



def get_shape_with_biggest_contours(selected_image):

    contours, hierarchy = cv2.findContours(selected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initializing
    biggest = np.array([])

    max_area = 0
    #

    for ch in contours:

        area = cv2.contourArea(ch)

        if area > 3000:

            cv2.drawContours(image1, ch, -1, (255,0,0), 5)  #################################

            arc_length = cv2.arcLength(ch, True)

            approx = cv2.approxPolyDP(ch, 0.02 * arc_length, True)

            if ( area > max_area ) and ( len(approx) == 4 ):

                biggest = approx

                max_area = area

    cv2.drawContours(image1, biggest, -1, (0,255,0), 12)  #################################

    return biggest



def reorder(points):

    # previously it was (4,1,2)
    points = points.reshape((4,2))

    # creating an empty array with the dimentions of (4,1,2)
    my_new_points = np.zeros((4,1,2), np.int32)

    # finding the smallest and points elements of the array
    sum_array = points.sum(1)

    my_new_points[0] = points[ np.argmin(sum_array) ]
    my_new_points[3] = points[ np.argmax(sum_array) ]

    # finding the other two elements of the array
    diff_array = np.diff(points, axis=1)

    my_new_points[1] = points[ np.argmin(diff_array) ]
    my_new_points[2] = points[ np.argmax(diff_array) ]

    return my_new_points



def getWarp(selected_image, biggest):

    print(biggest.shape)

    biggest = reorder(biggest)

    point_1 = np.float32(biggest)

    point_2 = np.float32( [[0,0], [w,0], [0,h], [w,h]] )

    matrix = cv2.getPerspectiveTransform(point_1, point_2)

    image_output = cv2.warpPerspective( selected_image, matrix, (w,h) )

    return image_output



video = cv2.VideoCapture(0)

w = 640
h = 480

video.set(3, w)

video.set(4, h)

video.set(10, 150)

while True:

    status, image1 = video.read()

    image_thresh = preProcessing(image1)

    biggest = get_shape_with_biggest_contours(image_thresh)

    print(biggest)


    cv2.imshow("Webcam1", image1)

    cv2.imshow("Webcam2", image_thresh)


    if biggest.size != 0:

        image_warp = getWarp(image1, biggest)

        new_image_warp = cv2.resize(image_warp, (h,w))

        cv2.imshow("Webcam4", new_image_warp)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break