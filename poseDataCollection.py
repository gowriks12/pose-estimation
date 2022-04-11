import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm
import csv


cTime = 0
pTime = 0
cap = cv2.VideoCapture(1)
detector = pm.PoseDetector()
counter = 0
classes = ['Attentive', 'Intermediate', 'Inattentive']
print(len(classes))
classNum = 0

with open('poselandmarks.csv', mode='w') as poseLandmark_file:
    landmark_writer = csv.writer(poseLandmark_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    landmark_writer.writerow(["Landmarks", "class"])
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)
        lmDist=[]
        lmDist.append(0)
        if(len(lmList) != 0):
            # print("in if", len(lmList))
            for i in range(1, len(lmList)):
                # print(i)
                lmDist.append(((lmList[i][0] - lmList[0][0])**2 + (lmList[i][1] - lmList[0][1])**2)**0.5)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(classes[classNum]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if counter == 30:
            counter = 0
            classNum += 1
            print(classes[classNum])

        elif classNum >= len(classes) or k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:
            # SPACE pressed
            # img_name = "opencv_frame_{}.png".format(img_counter)
            # cv2.imwrite(img_name, img)
            # print("{} written!".format(img_name))
            print(lmDist)
            landmark_writer.writerow([lmDist, classes[classNum]])
            counter += 1
            print(counter)

cap.release()
cv2.destroyAllWindows()