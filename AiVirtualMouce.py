import cv2
import numpy as np
import HandTrackingModule as htm
from cvzone.HandTrackingModule import HandDetector
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 8
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# detector = htm.handDetector(maxHands=1)
detector = HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # (x,y,z) List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
        x1,y1 = lmList1[8][:2]
        x2,y2 = lmList1[12][:2]
        tipOfIndexFinger = lmList1[8][0:2]
        tipOfMiddleFinger = lmList1[12][0:2]
        #print(x1,y1,x2,y2)


        fingers1 = detector.fingersUp(hand1)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)


        # print(fingers1)
        if fingers1[1]==1 and fingers1[2]==0:
            x3 = np.interp(x1, (frameR, wCam-frameR ), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR ), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1),15,(255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers1[1]==1 and fingers1[2]==1:
            length, info, img = detector.findDistance(tipOfIndexFinger,tipOfMiddleFinger, img, color=(255, 0, 0),scale=10)
            # print(info)
            if length<40:
                cv2.circle(img, (info[4], info[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    #display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
