from os import listdir
from os.path import isfile, join
from PIL import Image

import random
import math
import time
import numpy as np
import cv2
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

from ..pyTorchModels import steeringModel as steeringAngleModel

widthSteering = heightSteering = 140

widthRoad = 600
heightRoad = 200

modelsPath = 'Models'
steeringModel = 'steeringModel.torch'
pathToModel = f'{modelsPath}/{steeringModel}'

startingDrivingFramesPath = 'StartingDrivingFrames'
labledDrivingFramesPath = 'LabeledDrivingFrames'


#road crop area
x1R = 400
x2R = 600
y1R = 630
y2R = 1280

#road image size
widthRoad  = y2R - y1R
heightRoad = x2R - x1R

#wheel crop area
x1W = 812
x2W = 932
y1W = 913
y2W = 1033

#wheel image size
widthWheel  = y2W - y1W
heightWheel = x2W-x1W

def setUpPaths():
    exist = os.path.exists(modelsPath)
    if exist:
        print('Models Directoty Exists')
    else:
        print('Models Directory Does not Exists')
        print("Please Add Model For Steering Angle")
        input("Click any key to continue once Model added...........")

    exist = os.path.exists(startingDrivingFramesPath)
    if exist:
        print('Starting Frames Directoty Exists')
    else:
        os.makedirs(startingDrivingFramesPath)
        print('Starting Frames Directory Does not Exists')
        print("Please Add Starting Images For Driving Augmentation")
        input("Click any key to continue once images added...........")


    exist = os.path.exists(labledDrivingFramesPath)
    if not exist:
        os.makedirs(labledDrivingFramesPath)
        print("Labled Frames Directoty Created")
    else:
        print('Labled Frames Directory Exists, Deleting and Recreating')
        shutil.rmtree(labledDrivingFramesPath)
        os.makedirs(labledDrivingFramesPath)

def AugmentDrivingData():
    setUpPaths()

    files = [f for f in listdir(startingDrivingFramesPath) if isfile(join(startingDrivingFramesPath, f))]
    
    print("Augementation Starting")

    #load net
    model = steeringAngleModel.steerAngleModel

    # Set device GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # load model to GPU and set to eval mode
    model = model.to(device)
    model.load_state_dict(torch.load(pathToModel))
    model.eval()


    for file in files:
        print(file)
        noExtension = file[:-4]
        filePath = f'{startingDrivingFramesPath}/{file}'


        fileSize = os.path.getsize(filePath)
        if fileSize>100:
            #load image
            wholeFrame = cv2.imread(filePath)
            

            #wheel angle prediction
            crop = wholeFrame[x1W:x2W, y1W:y2W]
            grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            (thresh, contrastImg) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
            backtorgb = cv2.cvtColor(contrastImg,cv2.COLOR_GRAY2RGB)
            transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((heightWheel,widthWheel)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
            wheel=transformImg(np.array(backtorgb))
            wheel = wheel.to(device).unsqueeze(0)
            with torch.no_grad():
                prediction = model(wheel)
            detectedAngle = prediction.data.cpu().numpy()[0][0]
            realAngle = (180*detectedAngle)-90
            

            # crop to road
            crop = wholeFrame[x1R:x2R, y1R:y2R]

            # resize to 180*56
            resized = cv2.resize(crop, [180,56], interpolation = cv2.INTER_AREA)


            cv2.imwrite(f'{labledDrivingFramesPath}/{noExtension}_roadcrop_alpha_none_filter_none_angleNorm_{detectedAngle}.jpg', resized)

            #array of all angle values to adjust brightness by
            alphas = [0.2,0.4,0.6,0.8,1.2,1.4]
            
            for alpha in alphas:

                    brighness = cv2.convertScaleAbs(resized, alpha, alpha)
                    cv2.imwrite(f'{labledDrivingFramesPath}/{noExtension}_roadcrop_alpha_{alpha}_filter_none_angleNorm_{detectedAngle}.jpg', brighness)

                    for f in range(2,3):
                        blur = cv2.blur(brighness,(f,f))

                        cv2.imwrite(f'{labledDrivingFramesPath}/{noExtension}_roadcrop_alpha_{alpha}_filter_{f}_angleNorm_{detectedAngle}.jpg', blur)


    augmentedFrames = [f for f in listdir(labledDrivingFramesPath) if isfile(join(labledDrivingFramesPath, f))]
    print('Total Augmented Frames = ', len(augmentedFrames))