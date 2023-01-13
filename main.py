import numpy as np
import cv2
import json

import datetime
import imutils

from sideway_bounding_box import sideway_box
from click_sideways_picture import click_sideway_person_picture
from click_entry_pictures import click_entry_person_picture
from draw_bounding_box_on_people import draw_box_on_person
from entry_bounding_box import draw_entry_bounding_box



def main():
    f = open('cam_config.json')
    data = json.load(f)

    video = cv2.VideoCapture("vid.mp4")
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    # Only enable it if you are using OpenVino environment
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]


    # denormalized_x = x * w
    # denormalized_y = y * h
    total_frames = 0
    count = 0
    count_sideway = 0

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        # Read the frame
        ret, frame = video.read()
        frame = imutils.resize(frame, width=900)
        total_frames = total_frames + 1

        (height, width) = frame.shape[:2]

        # If the frame was not read properly, break the loop
        if not ret:
            break
        
        start_point1,end_point1 = sideway_box(data,frame,height,width)

        click_sideway_person_picture(frame,start_point1,end_point1,hog)


        draw_box_on_person(frame,width,height,detector,CLASSES)

        start_point2,end_point2 = draw_entry_bounding_box(data,frame,height,width)
        
        click_entry_person_picture(frame,start_point2,end_point2,hog)

        # Display the frame
        cv2.imshow("Video", frame)

        # Check if the user pressed the 'q' key
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()