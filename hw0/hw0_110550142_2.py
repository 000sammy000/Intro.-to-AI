import cv2
import imutils
import numpy as np

cap=cv2.VideoCapture("video.mp4")

object_detector=cv2.createBackgroundSubtractorMOG2()

fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

last=None
while True:
    ret, frame=cap.read()
    frame = cv2.resize(frame, (500, 281), interpolation=cv2.INTER_AREA)
    #frame1=frame[:,:,0]
    mask=object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
         
    img1 = cap.read()[1]
    #img2 = cap.read()[1]
    

    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1,(5,5),0)
    #blur2 = cv2.GaussianBlur(gray2,(5,5),0)

    if last is None:
        last=blur1
        
    result = cv2.absdiff(blur1, last)
    last=blur1
    
    result = cv2.resize(result, (500, 281), interpolation=cv2.INTER_AREA)
    ret, th = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, None, iterations=36)

    
    mog = fgbg_mog.apply(frame)
    mog = cv2.morphologyEx(mog, cv2.MORPH_OPEN, kernel)
    b = np.zeros(mog.shape[:2], dtype = "uint8")
    r = np.zeros(mog.shape[:2], dtype = "uint8")

    result_rgb=cv2.merge([b,result,r])

    stk=np.hstack((frame,result_rgb))
    #cv2.imshow("Frame",frame,width=600)
    #cv2.imshow("Mask",mask)
    cv2.imshow("Result",stk)
    #cv2.imshow("Result",result_rgb)

    key=cv2.waitKey(30)
    if key==27 or key==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
