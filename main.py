
import cv2 as cv
import numpy as np

#function for road lane detection on a single frame
def road_lane_for_image (image, pt1, pt2, pt3):
    #copy image
    copy = np.copy(image)

    #turn image to grayscale
    gray = cv.cvtColor(copy,cv.COLOR_BGR2GRAY)

    #applying a filter with a threshold to keep only the lighter pixels and turning the darker to full black
    # (to prevent the detection of non white edges with wouldn't correspond to road lines)
    for i in range (0, len(gray)) :
        for j in range(0, len(gray[0])) :
            if(gray[i][j]<(230)):
                gray[i][j]=0

    #blur image
    blured = cv.GaussianBlur(gray, (5,5), 0)

    #detect edges
    canny = cv.Canny(blured, 50, 150)

    #masking with a triangular shape
        #creating triangle
    blank = np.zeros(image.shape[:2], dtype='uint8')
    pts=np.array([pt1, pt2, pt3])

        #filling the triangle
    cv.fillPoly(blank, [pts], 255)

        #applying the mask
    masked = cv.bitwise_and(canny,blank)

    #finding the lines in the image
    lines = cv.HoughLinesP(masked, 4, np.pi / 180, 100, np.array([]), minLineLength=15, maxLineGap=5)

    # displaying the lines on a blank image
    def display_lines(image, lines) :
        line_image=np.zeros_like(image)
        if lines is not None :
            for line in lines :
                x1,y1,x2,y2 = line.reshape(4)
                cv.line(line_image, (x1,y1), (x2,y2), (255,255,255),2)
        return line_image

    line_image  = display_lines(masked, lines)

    # displaying the line on the original image
    for i in range (0, len(line_image)) :
        for j in range(0, len(line_image[0])) :
            if(line_image[i][j]!=(0)):
                copy[i][j]=[0,255,0]
    return copy

#test for single frame
image_test = cv.imread('raw_road_lane.jpg')
# creating the points for the triangle (depends on the camera of the car)
p1=[int(image_test.shape[1]/8),image_test.shape[0]]
p2=[int(image_test.shape[1]*9/10), image_test.shape[0]]
p3=[int(image_test.shape[1]/2), int(image_test.shape[0]*6.5/12)]
cv.imshow('result', road_lane_for_image(image_test, p1, p2, p3))


#test for a video
capture = cv.VideoCapture('Highway.mp4')
while (capture.isOpened()) :
    isTrue, frame = capture.read()
    # creating the points for the triangle (depends on the camera of the car)
    pt1 = [0, frame.shape[0]]
    pt2 = [int(frame.shape[1]), frame.shape[0]]
    pt3 = [int(frame.shape[1] *0.55), int(frame.shape[0] * 5.5 / 12)]
    frame_with_lanes = road_lane_for_image(frame,pt1,pt2,pt3)
    cv.imshow('Video', frame_with_lanes)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()

