import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    cap=cv2.VideoCapture("data/detect/video.gif")
    txt=[]
    while True:
        f = open(data_path, 'r')
        ret,frame=cap.read()
        if ret:
            n=int(f.readline())
            
            temp=[]
            for i in range(n):
                #print(n)
                L=f.readline()
                cr=list(map(int, L.split()))
                
                parkingspace=crop(cr[0],cr[1],cr[2],cr[3],cr[4],cr[5],cr[6],cr[7],frame)
                parkingspace=cv2.resize(parkingspace,(36,16), interpolation=cv2.INTER_AREA)
                img=cv2.cvtColor(parkingspace,cv2.COLOR_BGR2GRAY)
                parkingspace_test=[]
                parkingspace_test.append(np.ravel(img))
                car=clf.classify(parkingspace_test)
                temp.append(car)
                if car==1:
                    green_color = (0, 255, 0)
                    cv2.line(frame, (cr[0],cr[1]), (cr[2], cr[3]), green_color, 2)
                    cv2.line(frame, (cr[2],cr[3]), (cr[6], cr[7]), green_color, 2)
                    cv2.line(frame, (cr[6],cr[7]), (cr[4], cr[5]), green_color, 2)
                    cv2.line(frame, (cr[0],cr[1]), (cr[4], cr[5]), green_color, 2)
                


            txt.append(temp)
            cv2.imshow("show",frame)
            f.close()
            
            key=cv2.waitKey(1) & 0xff
            if key==ord('q'):
                break
        else:
            break

    path = 'ML_Models_pred.txt'
    with open(path, 'w') as f:
        for i in range(len(txt)):
            string=str(txt[i])
            string = string.replace("[","").replace("]","")
            f.write(string)
            f.write('\n')


            
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)
