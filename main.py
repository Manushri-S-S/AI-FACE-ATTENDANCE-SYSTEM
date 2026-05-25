import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime

path = 'images'

images = [] # Stores all student images loaded from the images folder
classNames = [] #gets the name from the images we stored (ex:name.jpeg)
last_marked = {} #last timestamp of login of each individual gets stored here
present_students = set() # Stores students detected during current session

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
all_students = set(name.upper() for name in classNames)

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):

    current_time = time.time()

    if name in last_marked:

        elapsed = current_time - last_marked[name]

        if elapsed < 60:
            return

    last_marked[name] = current_time

    with open('Attendance.csv', 'a+') as f:

        now = datetime.now()

        dtString = now.strftime('%d-%m-%Y %H:%M:%S')

        f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)

print("Attendance Marked")

# open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # compare current face with stored faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

       # print(faceDis[matchIndex])#shows the accuracy Smaller number = better match
        if matches[matchIndex]:
        
            name = classNames[matchIndex].upper()
            present_students.add(name)

            y1, x2, y2, x1 = faceLoc

            scale = 1 / 0.25

            y1, x2, y2, x1 = int(y1*scale), int(x2*scale), int(y2*scale), int(x1*scale)

            # GREEN rectangle for known face
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)

            cv2.putText(img, name, (x1+6,y2-6),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (255,255,255), 2)

            markAttendance(name)
        
        else:

            y1, x2, y2, x1 = faceLoc

            scale = 1 / 0.25

            y1, x2, y2, x1 = int(y1*scale), int(x2*scale), int(y2*scale), int(x1*scale)

            # RED rectangle for unknown face
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)

            cv2.putText(img, "UNKNOWN", (x1+6,y2-6),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (255,255,255), 2)

    cv2.imshow('Face Attendance', img)

    if cv2.waitKey(1) & 0xFF == ord('m'):#To stop press m on the display
        break

absent_students = all_students - present_students

cap.release()
print("\nABSENT STUDENTS:")
for student in absent_students:
    print(student)
cv2.destroyAllWindows()