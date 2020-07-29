# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:28:43 2020

@author: himad

the problem in using canny here will be it will detect the border of paper on 
which bar code is drawn.

"""

import cv2
import numpy as np

image = cv2.imread('bar4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gradX = cv2.Sobel(gray,cv2.CV_32F,  dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray,cv2.CV_32F,  dx=0, dy=1, ksize=-1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)


cv2.imshow("gradient", gradient)
blurred = cv2.blur(gradient, (9,9))
_, thresh = cv2.threshold(blurred, 224, 255, cv2.THRESH_BINARY )

cv2.imshow("blurred", blurred)
cv2.imshow("thresh", thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
#sometimes we need our kernel to be circle or something sowe use above function

closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
#The result will look like the outline of the object.

#remove extra noise
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
cv2.imshow("closed", closed)


#finding the contour and sort contour by area to keep the largest contour
cnts, hierarchyq = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)

c= sorted(cnts, key=cv2.contourArea, reverse=True)[0]

rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(image, [box], -1, (0,255, 0), 3)
cv2.imshow("image", image)


cv2.waitKey(0)
cv2.destroyAllWindows()