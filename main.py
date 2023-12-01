from ultralytics import YOLO
import keyboard

import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 5000)
cap.set(4, 6000)

model = YOLO('SignWeights3.pt')

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
word = ""
last = ""
classString = ""

while True:
    success, img = cap.read()

    results = model(img, stream=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            classInt = int(box.cls[0])
            classString = class_names[classInt]
            if last != classString:
                word = word + classString
    img = cv2.putText(img, word, (150, 250), cv2.FONT_HERSHEY_SIMPLEX , 5, (255,0,0), 2, cv2.LINE_AA)
    last = classString
    if keyboard.is_pressed('q'):
        word = ""
    img = cv2.resize(img, (0,0), None, 1.25,1.25)
    #img = cv2.flip(img, 1)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff == ord('c'):
        break