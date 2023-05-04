import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import shutil

labledSteeringFramesPath = "LabledSteeringFrames"
startingPath = "StartingSteeringFrames"

def setUpPaths():
    exist = os.path.exists(labledSteeringFramesPath)
    if not exist:
        os.makedirs(labledSteeringFramesPath)
        print("Directoty GeneratedImages Created")
    else:
        print('Directory GeneratedImages Exists, Removing and Recreating')
        #Uncomment these two lines if you want to delete the frames folder before adding more frames
        shutil.rmtree(labledSteeringFramesPath)
        os.makedirs(labledSteeringFramesPath)   

    
    exist = os.path.exists(startingPath)
    if not exist:
        os.makedirs(startingPath)
        print("Directoty Created")
        print("Please Add Starting Image For Steering Augmentation")
        input("Click any key to continue once images added...........")
    else:
        print('Directory Exists')

def AugmentSteeringData():
    setUpPaths()

    files = [f for f in listdir(startingPath) if isfile(join(startingPath, f))]
    
    if len(files) == 0:
        print("Please Add Starting Image For Steering Augmentation")
        input("Click any key to continue once images added...........")
    
    print("Augementation Starting")
    
    for file in files:
        print(file)
        #dont want the '.jpg' element of the file name in the file name
        noExtension = file[:-4]

        #load image
        img = cv2.imread(f'{startingPath}/'+file)
        xA = 812
        xB = 932
        yA = 913
        yB = 1033

        #crop to steering wheel logo
        crop = img[xA:xB, yA:yB]

        grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
        (thresh, contrastImg) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)

        backtorgb = cv2.cvtColor(contrastImg,cv2.COLOR_GRAY2RGB)


        #save crop
        cv2.imwrite(f'GeneratedImages/{noExtension}_0_degrees_x0y0_crop.jpg',backtorgb)
        
        #array of all angle values to rotate by
        angles = np.linspace(-90,90,180)
        
        
        for i in range(-8,8):
            for j in range(-20,10):
                for a in angles:
                    angle = round(a)

                    #finder center of image
                    (h, w) = img.shape[:2]
                    (cX, cY) = (w // 2, h // 2)
                    
                    # rotate our image by angle a
                    M = cv2.getRotationMatrix2D((971, 894), angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h))
                    
                    #crop to steering wheel logo
                    x1 = xA + i
                    x2 = xB + i
                    y1 = yA + j
                    y2 = yB + j
                    crop = rotated[x1:x2, y1:y2]

                    grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
                    (thresh, contrastImg) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)

                    backtorgb = cv2.cvtColor(contrastImg,cv2.COLOR_GRAY2RGB)

                    # ground truth
                    # scale angle between 0 and 1
                    angle = angle + 90
                    angle = round((angle-(0))/(180-(0)),4)

                    #save image
                    cv2.imwrite(f'{labledSteeringFramesPath}/{noExtension}_{angle}_degrees_x{i}y{j}_crop.jpg',backtorgb)