import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm
import pickle

# load the model from disk
# loaded_model = pickle.load(open('knnpickle_file', 'rb'))
loaded_model = pickle.load(open('svm_pickle_file.sav', 'rb'))
# result = loaded_model.predict([test])

cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    image = detector.findPose(img)
    className = ""
    lmList = detector.findPosition(img,draw=False)
    lmDist = []
    lmDist.append(0)
    if len(lmList) != 0:
        # Calculating the distance between points
        for i in range(1, len(lmList)):
            lmDist.append(((lmList[i][0] - lmList[0][0])**2 + (lmList[i][1] - lmList[0][1])**2)**0.5)
        # Predict gesture
        prediction = loaded_model.predict([lmDist])
        # print(prediction)
#             classID = np.argmax(prediction)
        className = prediction[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

