import cv2
import os
import time

count2 = 1
def click_entry_person_picture(frame,start_point2,end_point2,hog):
    global count2

    entry_frame = frame[end_point2[1]:start_point2[1],start_point2[0]:end_point2[0]]
    (boxes, weights) = hog.detectMultiScale(entry_frame, winStride=(4,4))
    print(boxes)
    for (x, y, w, h) in boxes:
        person = entry_frame[y:y+h, x:x+w]
        print(person)
        os.makedirs("captured_images/entry",exist_ok=True)
        cv2.imwrite(f"captured_images/entry/{count2}.jpg",person)
        count2 += 1
        print(f"sideway2 is {count2}")

    