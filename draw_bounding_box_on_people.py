import cv2
import numpy as np

def draw_box_on_person(frame,width,height,detector,CLASSES):
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