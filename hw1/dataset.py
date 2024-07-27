import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    path1=data_path+'/car'
    dataset=[]
    
    for filename in os.listdir(path1):
 
        if filename.endswith('.png'):
            image=cv2.imread(os.path.join(path1,filename))
            image=cv2.resize(image,(36,16), interpolation=cv2.INTER_AREA)
            img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            label=1

            dataset.append((img,label))
        
    path2=data_path+'/non-car'

    
    
    for filename in os.listdir(path2):
    
        if filename.endswith('.png'):
            image=cv2.imread(os.path.join(path2,filename))
            image=cv2.resize(image,(36,16), interpolation=cv2.INTER_AREA)
            img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    
            label=0

            dataset.append((img,label))
                
    
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset


