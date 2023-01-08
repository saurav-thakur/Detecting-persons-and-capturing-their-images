import cv2

def sideway_box(data,frame,height,width):
    sideway = data['siteConfig']['cameraConfig']['201']['roi'][0]['coordinates']
    start_point1 = (int(sideway[2]['x']*width), int(sideway[2]['y']*height)) # denormalizing the coordinates
    end_point1 = (int(sideway[1]['x']*width), int(sideway[1]['y']*height))
    cv2.rectangle(frame, start_point1, end_point1, (255, 0, 0), 2)

    return start_point1,end_point1