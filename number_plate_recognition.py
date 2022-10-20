## This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import imutils
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

#import glob

#path = glob.glob("C:/Users/Admin/PycharmProjects/pythonProject1/cars/*.jpg")

#for file in path:
    # print(file)
#    img = cv2.imread(file)
#    cv2.imshow('Image', img)
#    cv2.waitKey(0)

    # If you don't have tesseract executable in your PATH, include the following
    #Python-tesseract is an optical character recognition (OCR) tool for python.
    # That is, it will recognize and “read” the text embedded in images.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('car.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
img = cv2.resize(img, (600, 400))
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #A bilateral filter is a noise-reducing smoothing filter for images.
    # It replaces the intensity of each pixel with a weighted average of intensity values
    # from nearby pixels. This weight can be based on a Gaussian distribution
    #The bilateral filter smooths an input image while preserving its edges
gray = cv2.bilateralFilter(gray, 13, 17, 17)
    #d: Diameter of each pixel neighborhood. Then,
    #The greater the value, the colors farther to each other will start to get mixed.



    #The Canny edge detector is an edge detection operator that uses
    # a multi-stage algorithm to detect a wide range of edges in images.
edged = cv2.Canny(gray, 30,200)


    #The contours are a useful tool for shape analysis and object detection and recognition.
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    #Imutils is a package based on OpenCV, which can call the opencv interface more simply.
    # It can easily realize a series of operations such as image translation, rotation, scaling,
contours = imutils.grab_contours(contours) #contour is boundary of an ibject
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour,10,True)
    if len(approx) == 4:
        location = approx
        break
print(location)


mask = np.zeros(gray.shape, np.uint8) #created a blank mask
new_image = cv2.drawContours(mask,[location],0,255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)

#plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY))


(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
Cropped = gray[x1:x2 + 1, y1:y2 + 1]

text = pytesseract.image_to_string(Cropped, config="--psm 6")
print("Detected license plate Number is:", text)








#code for 2nd car
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread('car2.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
#img = cv2.resize(img, (600, 400))
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 15, 17, 17)
edged = cv2.Canny(gray, 30,200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours) #contour is boundary of an ibject
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour,10,True)
    if len(approx) == 4:
        location = approx
        break
print(location)


mask = np.zeros(gray.shape, np.uint8) #created a blank mask
new_image = cv2.drawContours(mask,[location],0,255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
Cropped = gray[x1:x2 + 1, y1:y2 + 1]

text = pytesseract.image_to_string(Cropped, config="--psm 6")
print("Detected license plate Number is:", text)









#3rd car
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread('car3.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
#img = cv2.resize(img, (600, 400))
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 15, 17, 17)
edged = cv2.Canny(gray, 30,200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours) #contour is boundary of an ibject
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour,10,True)
    if len(approx) == 4:
        location = approx
        break
print(location)


mask = np.zeros(gray.shape, np.uint8) #created a blank mask
new_image = cv2.drawContours(mask,[location],0,255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
Cropped = gray[x1:x2 + 1, y1:y2 + 1]

text = pytesseract.image_to_string(Cropped, config="--psm 6")
print("Detected license plate Number is:", text)































