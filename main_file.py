import numpy as np
import cv2
import json

import datetime
import imutils

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

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Read the frame
    ret, frame = video.read()
    frame = imutils.resize(frame, width=600)
    total_frames = total_frames + 1

    (height, width) = frame.shape[:2]

    # If the frame was not read properly, break the loop
    if not ret:
        break
    
    sideway = data['siteConfig']['cameraConfig']['201']['roi'][0]['coordinates']
    start_point1 = (int(sideway[2]['x']*width), int(sideway[2]['y']*height)) # denormalizing the coordinates
    end_point1 = (int(sideway[1]['x']*width), int(sideway[1]['y']*height))
    cv2.rectangle(frame, start_point1, end_point1, (255, 0, 0), 2)

    sideway_frame = frame[end_point1[1]:start_point1[1],start_point1[0]:end_point1[0]]
    (boxes, weights) = hog.detectMultiScale(sideway_frame, winStride=(4, 4))

    for (x, y, w, h) in boxes:
        person = sideway_frame[y:y+h, x:x+w]
        cv2.imwrite('./captured_images/person_{}.jpg'.format(count), person)
        count += 1
    # # width1 = 964-57
    width1 = int(sideway[1]['x']*width) - int(sideway[0]['x']*width)

    # # height1 = 994-447
    height1 = int(sideway[2]['y']*height) - int(sideway[1]['y']*height)
    capture_frame = frame[start_point1[1]:end_point1[1],start_point1[0]:end_point1[1]]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()


   
    # print(width1,height1)

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)


    entry = data['siteConfig']['cameraConfig']['201']['roi'][1]['coordinates']
    start_point2 = (int(entry[2]['x']*width), int(entry[2]['y']*height))
    end_point2 = (int(entry[1]['x']*width), int(entry[1]['y']*height))
    cv2.rectangle(frame, start_point2, end_point2, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Check if the user pressed the 'q' key
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()