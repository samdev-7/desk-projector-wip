import cv2
import os 

os.environ['DISPLAY'] = '192.168.0.32:10.0'
cv2.namedWindow("Debug")

img = cv2.imread('Debug_screenshot_05.08.2024.png')

cv2.imshow("Debug", img)

blur = cv2.GaussianBlur(img, (3, 3), 0)