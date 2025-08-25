import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import Tracker
import numpy as np
import cvzone



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


model = YOLO("best.pt")  

cap=cv2.VideoCapture('cow2.mp4')
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0
tracker=Tracker()
cowcount=[]
cy1=487
offset=6

while True:
    ret,frame = cap.read()
    
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    
    frame = cv2.resize(frame, (1008, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        list.append([x1,y1,x2,y2])
    bbox_list=tracker.update(list)
    for bbox in bbox_list:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
           cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
           cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
           if cowcount.count(id)==0:
               cowcount.append(id)
            
            
    
               
                                   
    counting=len(cowcount)
    cvzone.putTextRect(frame,f'{counting}',(50,60),2,2)
    cv2.line(frame,(5,487),(1019,487),(255,0,255),2)
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
