#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

video_path="/home/anusaini/pytorch_projects/ML-ASSIGNMENT/task_2_video.mp4"
    
video = cv2.VideoCapture(video_path)
cc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = video.get(cv2.CAP_PROP_FPS)
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_file = cv2.VideoWriter('output.mp4', cc, 200, (width, height))
x=0
while x==0:
    _,frame = video.read()  
    if not _:
        x=1
        break
    # blurring the frame
    frame_gaussian_blur = cv2.GaussianBlur(frame, (3, 3), 0)
    # converting BGR to HSV
    hsv = cv2.cvtColor(frame_gaussian_blur, cv2.COLOR_BGR2HSV)
    # the range of green color in HSV
    lower_green = np.array([32, 25, 36])
    higher_green = np.array([80, 255, 255])
    # getting the range of green color in videoframe
    green_range = cv2.inRange(hsv, lower_green, higher_green)
    # getting the gray channel
    green_gray = cv2.GaussianBlur(green_range, (3, 3), 0)
    # applying HoughCircles
    rows = green_gray.shape[0]
    detected_circles = cv2.HoughCircles(green_gray, cv2.HOUGH_GRADIENT, 1, rows/4, 80, 30, 20, 0)
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0,:]:
        # drawing on detected circle and its center
        cv2.circle(frame,(pt[0],pt[1]),3,(0,0,255),5)
        cv2.imshow('detected_circles', frame)
        output_file.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
video.release()
output_file.release()
cv2.destroyAllWindows()

