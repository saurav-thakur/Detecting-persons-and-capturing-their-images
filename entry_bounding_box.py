import cv2

def draw_entry_bounding_box(data,frame,height,width):
    entry = data['siteConfig']['cameraConfig']['201']['roi'][1]['coordinates']
    start_point2 = (int(entry[2]['x']*width), int(entry[2]['y']*height))
    end_point2 = (int(entry[1]['x']*width), int(entry[1]['y']*height))
    cv2.rectangle(frame, start_point2, end_point2, (255, 0, 0), 2)

    return start_point2,end_point2