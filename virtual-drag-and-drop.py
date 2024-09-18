import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHand = mp.solutions.hands
hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

cx , cy , w , h = 100, 100, 200, 200

colorR = 0 , 0 , 255


while True:
    _ , img = cap.read()
    img = cv2.flip(img , 1)
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    lmList = []
    
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img , handLms , mpHand.HAND_CONNECTIONS)
        
        for id ,lm in enumerate(handLms.landmark):
            h , w , _ = img.shape
            cx , cy = int(lm.x * w) , int(lm.y * h)
            lmList.append([id ,cx ,cy])
    
    if lmList:

        cursor = lmList[8]
        
        if cx- w // 2 < cursor[1] < cx+w//2 and \
            cy-h//2 < cursor[2] < cy+h//2:
            colorR = (0 , 255 , 0)
            cx , cy = cursor[1] , cursor[2]
        
        else:
            colorR = (0 , 0 , 255)



    cv2.rectangle(img , (cx-w//2 , cy-h//2) , (cx+w//2 , cy+h//2) , colorR , cv2.FILLED)
    
    
    
    
    
    
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)