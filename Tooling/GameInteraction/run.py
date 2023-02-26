import numpy as np
import cv2
from mss import mss
from PIL import Image

import os
 


#import pyautogui
import time

import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

modelsPath = 'Models'
modelName = 'bestModel.torch'
pathToModel = f'{modelsPath}/{modelName}'

#wheel crop area
x1W = 812
x2W = 932
y1W = 913
y2W = 1033

#wheel image size
widthWheel  = y2W - y1W
heightWheel = x2W-x1W

#area of screen that window will be present in
bounding_box = {'top': 340, 'left': 1490, 'width': 1920, 'height': 1080}

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc = nn.Linear(512, 128)
        
        self.branch_a1 = nn.Linear(128, 32)
        self.branch_a2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc(x))

        a = F.leaky_relu(self.branch_a1(x))
        
        out1 = self.branch_a2(a)
        
        return out1



def directionToSteer(current,target):
    difference = current-target
    #if difference > 0:
        #pyautogui.typewrite('a')
    #elif difference < 0:
        #pyautogui.typewrite('d')
    
def angleFormater(toFormat):
    missing = 4-len(str(toFormat))
    if toFormat >=0:
        return ''.join([' ']*missing) + str(toFormat)
    if toFormat < 0:
        return ''.join([' ']*missing) + str(toFormat)

def printConsoleDebug(of,tpc,csa,tsa):
    of = "{:.3f}".format(of)
    tpc = "{:.3f}".format(tpc)
    csa = angleFormater(round(csa))
    tsa = angleFormater(round(tsa))
    os.system('clear')
    print(f'+-----------------------------------------------------------------------------+')
    print(f'| Euro Truck Driving Simulator Self Driving System          Version: 0.05     |')
    print(f'|-------------------------------+---------------------------------------------+')
    print(f'| Operating Freq:     {of}hz   | Current Steering Angle:    {csa}   Degrees   |')
    print(f'| Time Per Cycle:     {tpc}s    | Target Steering Angle:     {tsa}  Degrees    |')
    print(f'+-------------------------------+---------------------------------------------+')




if __name__ == "__main__":
    # Clearing the Screen
    os.system('clear')

    #load net
    #To Load Pretrained Weights:   weights='ResNet18_Weights.DEFAULT'
    resnet18 = torchvision.models.resnet18()
    resnet18.fc = nn.Identity()
    net_add=net()
    model = nn.Sequential(resnet18, net_add)

    # Set device GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # load model to GPU
    model = model.to(device)

    model.load_state_dict(torch.load(pathToModel))

    model.eval()

    sct = mss()

    print("Starting in 10 Seconds.........")
    time.sleep(10)

    #pyautogui.typewrite('1')

    startTime = time.time()

    while True:
        start = time.time()
        time.sleep(0.75)
        sct_img = sct.grab(bounding_box)
        wholeFrame = np.array(sct_img)

        #crop to steering wheel logo and transform image to black and white but in RGB color space
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

        targetAngle = 90

        #directionToSteer(realAngle,targetAngle)

        if time.time()-startTime > 30:
            cv2.destroyAllWindows()
            break
        
        elapsed = time.time()-start
        freq = 1/elapsed
        printConsoleDebug(freq,elapsed,realAngle,targetAngle)
