import cv2
import numpy as np
import os
import uuid

count = 1

def click_sideway_person_picture(frame,start_point1,end_point1,hog):
    global count
    sideway_frame = frame[end_point1[1]:start_point1[1],start_point1[0]:end_point1[0]]
    (boxes, weights) = hog.detectMultiScale(sideway_frame, winStride=(8, 8))
    

    for (x, y, w, h) in boxes:
        person = sideway_frame[y:y+h, x:x+w]
        print(person)
        os.makedirs("captured_images/sideway",exist_ok=True)
        cv2.imwrite(f"captured_images/sideway/{count}.jpg",person)
        count += 1
        print(f"sideway is {count}")


