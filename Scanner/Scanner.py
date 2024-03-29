import numpy as np
import cv2

def pre_processing(selected_image):
    image_gray = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)
    image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image_blur = cv2.GaussianBlur(image_thresh, (5, 5), 1)
    image_canny = cv2.Canny(image_blur, 100, 200) # may need fine-tuning
    kernel = np.ones((5, 5))
    image_dilation = cv2.dilate(image_canny, kernel, iterations=2)
    image_erode = cv2.erode(image_dilation, kernel, iterations=1)
    return image_erode

def get_shape_with_biggest_contours(selected_image, image1):
    contours, _ = cv2.findContours(selected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest, max_area = np.array([]), 0

    for contour in contours:
        area = cv2.contourArea(contour)
        
        print(area)
        
        if  (area > 1600000): # may need fine-tuning
            arc_length = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
            
            if (area > max_area) and (len(approx) == 4):
                biggest = approx
                max_area = area

        cv2.drawContours(image1, biggest, -1, (0,255,0), 12)
    return biggest

def reorder(points):
    points = points.reshape((4,2))
    sum_array = points.sum(1)
    my_new_points = np.zeros((4,1,2), np.int32)
    my_new_points[0] = points[ np.argmin(sum_array) ]
    my_new_points[3] = points[ np.argmax(sum_array) ]
    diff_array = np.diff(points, axis=1)
    my_new_points[1] = points[ np.argmin(diff_array) ]
    my_new_points[2] = points[ np.argmax(diff_array) ]
    return my_new_points

def getWarp(selected_image, biggest, weidth, height):
    biggest = reorder(biggest)
    point_1 = np.float32(biggest)
    point_2 = np.float32([[0,0], [weidth,0], [0,height], [weidth,height]])
    matrix = cv2.getPerspectiveTransform(point_1, point_2)
    image_output = cv2.warpPerspective(selected_image, matrix, (weidth,height))
    return image_output

def main():
    video = cv2.VideoCapture(0)
    weidth, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.set(3, weidth)
    video.set(4, height)
    video.set(10, 200) # Adjust camera brightness as necessary

    while True:
        status, image1 = video.read()
        if not status:
            raise Exception("Image not read successfully")
        
        image_thresh = pre_processing(image1)
        biggest = get_shape_with_biggest_contours(image_thresh, image1)

        cv2.imshow("Original", image1)
        cv2.imshow("Processed", image_thresh)

        if biggest.size != 0:
            image_warp = getWarp(image1, biggest, weidth, height)
            image_warp_rotated = cv2.rotate(image_warp, cv2.ROTATE_90_CLOCKWISE) # Rotate the scanned image to correct its orientation
            cv2.imshow("Scanned", image_warp_rotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
    