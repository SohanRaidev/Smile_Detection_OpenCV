import cv2
import numpy as np
import random


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def party_popper_effect(frame):
    h, w, _ = frame.shape
    for i in range(100): 
        confetti_x = random.randint(0, w)
        confetti_y = random.randint(0, int(h)) 
        confetti_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(frame, (confetti_x, confetti_y), random.randint(2, 5), confetti_color, -1)


video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()

    if not ret:
        break  

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  
        minNeighbors=5,   
        minSize=(30, 30)  
    )

    smile_detected = False  

   
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.8,  
            minNeighbors=20,   
            minSize=(25, 25)   
        )

       
        for (sx, sy, sw, sh) in smiles:
            
            cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            smile_detected = True 

    
    if smile_detected:
        cv2.putText(frame, "Keep smiling!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
       
        party_popper_effect(frame)
    else:
        cv2.putText(frame, "Smile please!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

   
    cv2.imshow('Smile Detector', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
