from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3


def speak(command):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Giọng nam hoặc nữ
    engine.say(command)
    engine.runAndWait()


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/name.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame =  video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        crop_image =  frame[y:y+h, x:x+w, : ]
        
        resized_img =  cv2.resize(crop_image, (50, 50))
        
        img_new = resized_img.copy().flatten().reshape(1, -1)

        pre_name = knn.predict(img_new)

        cv2.putText(frame, pre_name[0], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")

        exist =os.path.isfile("Attendance/Attendance_" + date + ".csv")


        attendance = [str(pre_name[0]), str(timestamp)]


    cv2.imshow("frame", frame)

    k=cv2.waitKey(1)
    if k == ord("o"):
         if exist:
              with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                   writer =  csv.writer(csvfile)
                   writer.writerow(attendance)
              csvfile.close()
         else:
              with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                   writer =  csv.writer(csvfile)
                   writer.writerow(COL_NAMES)
                   writer.writerow(attendance)
              csvfile.close()
         speak("SIGNIN SUCCESS")
    if k == ord("q") :
        break



video.release()
cv2.destroyAllWindows()


