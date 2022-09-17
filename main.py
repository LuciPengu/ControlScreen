
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
from threading import Thread
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:

                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])

            classID = np.argmax(prediction)
            className = classNames[classID]

            points = result.multi_hand_landmarks[0].landmark
            finger = points[8]
            cord = _normalized_to_pixel_coordinates(finger.x,finger.y,frame.shape[1],frame.shape[0])
            x, y = 0, 0

            if cord != None:
                x, y = cord
                x = x - frame.shape[0]/2
                y = y - frame.shape[1]/2
            cv2.circle(frame, (cord), 10, (0,255,0), cv2.FILLED)
            def func1():
                pyautogui.moveRel(x/5, y/5, duration = 0.1)
            if className == "peace":
                Thread(target = func1).start()
            if className == "fist":
                pyautogui.click()
                time.sleep(3)

    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()