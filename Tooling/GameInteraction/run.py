import numpy as np
import cv2
from mss import mss
from PIL import Image

import os
 
import pyautogui
from pynput.keyboard import Key, Controller

import time

import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

modelsPath = 'Models'
steerAngleModelPath = f'{modelsPath}/steerAngleModel.torch'
driverModelPath = f'{modelsPath}/driverModel.torch'

#wheel crop area
x1W = 812
x2W = 932
y1W = 913
y2W = 1033

#wheel image size
widthWheel  = y2W - y1W
heightWheel = x2W-x1W

#refresh time
refreshRate = 0.05

#run time
stoppingTime = 60

#area of screen that window will be present in
bounding_box = {'top': 340, 'left': 1490, 'width': 1920, 'height': 1080}

#road crop
#road crop area
x1R = 400
x2R = 600
y1R = 630
y2R = 1280
roadWidth = 180
roadHeight = 56

class netWheel(nn.Module):
    def __init__(self):
        super(netWheel, self).__init__()
        self.fc = nn.Linear(512, 128)
        
        self.branch_a1 = nn.Linear(128, 32)
        self.branch_a2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        a = F.leaky_relu(self.branch_a1(x))
        out1 = self.branch_a2(a)
        return out1

class netNvidia(nn.Module):
    def __init__(self):
        super(netNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,24,5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24,36,5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36,48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48,64,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding=1),
            #nn.Flatten(),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*4*19, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, input):
        input = input.view(input.size(0), 3, roadHeight, roadWidth)
        output = self.conv_layers(input)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


def directionToSteer(current,target):
    difference = current-target
    if difference > 0.05:
        keyboard.press('d')
        time.sleep(0.005)
        keyboard.release('d')
        return 'd   '
    elif difference < -0.05:
        keyboard.press('a')
        time.sleep(0.005)
        keyboard.release('a')
        return 'a   '
    else:
        return None

    
def angleFormater(toFormat):
    missing = 4-len(str(toFormat))
    if toFormat >=0:
        return ''.join([' ']*missing) + str(toFormat)
    elif toFormat < 0:
        return ''.join([' ']*missing) + str(toFormat)

def printConsoleDebug(of,tpc,csa,tsa,kp):
    of = "{:.3f}".format(of)
    tpc = "{:.3f}".format(tpc)
    csa = angleFormater(round(csa))
    tsa = angleFormater(round(tsa))
    os.system('clear')
    print(f'+-----------------------------------------------------------------------------+')
    print(f'| Euro Truck Driving Simulator Self Driving System          Version: 0.06     |')
    print(f'|-------------------------------+---------------------------------------------+')
    print(f'| Operating Freq:     {of}hz  | Current Steering Angle:  {csa}   Degrees     |')
    print(f'| Time Per Cycle:     {tpc}s    | Target Steering Angle:   {tsa}   Degrees     |')
    print(f'|                               | Key Pressed This Cycle:     {kp}            |')
    print(f'+-------------------------------+---------------------------------------------+')




if __name__ == "__main__":
    # Clearing the Screen
    os.system('clear')

    #load net
    #To Load Pretrained Weights:   weights='ResNet18_Weights.DEFAULT'
    resnet18 = torchvision.models.resnet18()
    resnet18.fc = nn.Identity()
    net_add=netWheel()
    steerAngleModel = nn.Sequential(resnet18, net_add)

    #load driver net
    driverModel = netNvidia()

    # Set device GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # load models to GPU
    steerAngleModel = steerAngleModel.to(device)
    driverModel = driverModel.to(device)

    steerAngleModel.load_state_dict(torch.load(steerAngleModelPath))
    steerAngleModel.eval()

    driverModel.load_state_dict(torch.load(driverModelPath))
    driverModel.eval()

    sct = mss()

    keyboard = Controller()

    print("Starting in 10 Seconds.........")
    time.sleep(10)

    #pyautogui.typewrite('1')

    startTime = time.time()

    while True:
        start = time.time()
        
        sct_img = sct.grab(bounding_box)
        wholeFrame = np.array(sct_img)

        #crop to steering wheel logo and transform image to black and white but in RGB color space
        cropWheel = wholeFrame[x1W:x2W, y1W:y2W]
        grayImage = cv2.cvtColor(cropWheel, cv2.COLOR_BGR2GRAY)
        (thresh, contrastImg) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
        backtorgb = cv2.cvtColor(contrastImg,cv2.COLOR_GRAY2RGB)

        #convert image to tensor and send to gpu
        transformImg = tf.Compose([tf.ToPILImage(),tf.Resize((heightWheel,widthWheel)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        wheel = transformImg(np.array(backtorgb))
        wheel = wheel.to(device).unsqueeze(0)

        # make wheel angle prediction
        with torch.no_grad():
            prediction = steerAngleModel(wheel)

        realAngle = prediction.data.cpu().numpy()[0][0]
        realAngle = (180*realAngle)-90

        # crop road from full frame and resize
        cropRoad = wholeFrame[x1R:x2R, y1R:y2R]
        resizedRoad = cv2.resize(cropRoad, [180,56], interpolation = cv2.INTER_AREA)
        resizedRoad = resizedRoad[:,:,:3]

        #transform to tensor and send to device
        transformImg = tf.Compose([tf.ToPILImage(),tf.Resize((roadHeight,roadWidth)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
        resizedRoad = transformImg(np.array(resizedRoad))
        resizedRoad = resizedRoad.to(device).unsqueeze(0)

        # predict angle to steer at
        with torch.no_grad():
            prediction = driverModel(resizedRoad)

        targetAngle = prediction.data.cpu().numpy()[0][0]

        # real angle un normalise
        targetAngle = (180*targetAngle)-90

        # steer vehicle
        keyPressed = directionToSteer(realAngle,targetAngle)

        # ends program after set time
        if time.time()-startTime > stoppingTime:
            break
        
        # waits until next cycle is due
        wait = abs(refreshRate-(time.time()-start))
        if wait > 0:
            time.sleep(wait)
        
        # prints debug for user
        elapsed = time.time()-start
        freq = 1/elapsed
        printConsoleDebug(freq, elapsed, realAngle, targetAngle, keyPressed)

        