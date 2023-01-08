import cv2

def click_sideway_person_picture(frame,start_point1,end_point1,hog):
    count = 0
    
    sideway_frame = frame[end_point1[1]:start_point1[1],start_point1[0]:end_point1[0]]
    (boxes, weights) = hog.detectMultiScale(sideway_frame, winStride=(4, 4))

    for (x, y, w, h) in boxes:
        person = sideway_frame[y:y+h, x:x+w]
        cv2.imwrite('./captured_images/sideways_captured/person_{}.jpg'.format(count), person)
        count += 1

    