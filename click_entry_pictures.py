import cv2

def click_entry_person_picture(frame,start_point2,end_point2,hog):
    count = 0
    entry_frame = frame[end_point2[1]:start_point2[1],start_point2[0]:end_point2[0]]
    (boxes, weights) = hog.detectMultiScale(entry_frame, winStride=(4, 4))

    for (x, y, w, h) in boxes:
        person = entry_frame[y:y+h, x:x+w]
        cv2.imwrite('./captured_images/sideways_captured/person_{}.jpg'.format(count), person)
        count += 1

    