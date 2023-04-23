from os import listdir
from os.path import isfile, join
from PIL import Image

import random
import math
import time
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torchvision.models.segmentation
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

from ..pyTorchModels import steeringModel as steeringAngleModel
from ..dataTooling import steeringData as steeringData

from ..pyTorchModels import drivingModel as drivingModel
from ..dataTooling import drivingData as drivingData


Learning_Rate = 0.00002
width = 180
height = 56
batchSize = 1024
epochs = 1500

trainPercentage = 0.85
testPercentage = 0.10
valPercentage = 0.05

modelPath = 'Models/drivingModel.torch'

path = 'LabeledDrivingFrames/'
dataFiles = []
dataAngles = []


# Save data from training for graphs
losses=np.zeros([epochs]) 
mseAtEpochForVal = []
epochArr = []

def LoadImg(path):
    image = Image.open(path)
    transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    image=transformImg(np.array(image))
    return image

class DrivingData(Dataset):
    def __init__(self):
        # data loaidng
        self.files = dataFiles
        self.angles = dataAngles
        self.length = len(dataAngles)

    def __getitem__(self, index):
        imagePath = self.files[index]
        angle = self.angles[index]
        imageTensor = LoadImg(imagePath)
        return imageTensor, float(angle)
    
    def __len__(self):
        return self.length

def mseTest(dataloader, model, device):
        errorSquared = []

        # Set model to eval mode
        model = model.eval()
        msePerBatch = []

        for i, (images, angles) in enumerate(dataloader):
            # Load image and send to device
            images = images.to(device)

            angles = angles.unsqueeze(1)

            with torch.no_grad():
                prediction = model(images)  # Run net

            predAngles = prediction.data.cpu().numpy()

            mse = np.square(np.subtract(angles,predAngles)).mean()
            msePerBatch.append(mse)

        mse = sum(msePerBatch)/len(msePerBatch)

        return mse.item()

def train():
    exist = os.path.exists('Models')
    if not exist:
        os.makedirs('Models')
        print("Directoty Models Created")
    else:
        print('Models directory exists')

    for file in [f for f in listdir(path) if isfile(join(path, f))]:
        angle = file.split('_')[-1][:-4]
        dataFiles.append(path + file)
        dataAngles.append(angle)

    if len(dataFiles) != len(dataAngles):
        raise Exception('Length of file and angle arrays not equal')
    
    fullDataset = DrivingData()

    trainDataset, valDataset, testDataset = torch.utils.data.random_split(fullDataset, [trainPercentage, valPercentage, testPercentage])

    trainDataloader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    trainIter = iter(trainDataloader)

    valDataloader = DataLoader(dataset=valDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    valIter = iter(valDataloader)

    testDataloader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    testIter = iter(testDataloader)

    totalBatches = math.ceil(len(trainDataset)/batchSize)

    data = next(trainIter)
    imageTensor, angle = data
    print(len(imageTensor), len(angle), totalBatches)

    

    model = drivingModel.driverModel

    # Set device GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    # load model to GPU
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),lr=Learning_Rate) 

    tStart = time.time()
    saveStr = ''

    for epoch in range(epochs):
        print(f'Starting Epoch: {str(epoch + 1)}')
        t = time.time()
        batchTimes=[]
        for i, (images, angles) in enumerate(trainDataloader):
            # start batch timer
            batchStart = time.time()

            # Load image and send to device
            images = images.to(device)

            # Load GT and send to device
            angles = angles.to(device)

            #unqueeze angles to match return array from prediction
            angles = angles.unsqueeze(1)

            # set model to training mode
            model = model.train()

            # Make prediction
            predLevels = model(images)
            model.zero_grad()

            # Calculate loss
            loss = torch.nn.L1Loss()
            output = loss(predLevels, angles)

            # Backpropogate loss
            output.backward()

            # Apply gradient descent change to weight
            optimizer.step() 

            #append TTR batch
            batchTimes.append(time.time()-batchStart)

        #calculate val set MSE
        mseForEpoch = mseTest(valDataloader, model, device)
        
        epochArr.append(epoch)

        # Save loss
        losses[epoch]=output.data.cpu().numpy()

        #compare model against previous MSE values, save model if better than previous best
        if epoch > 0:
            if mseForEpoch < min(mseAtEpochForVal):
                saveStr = f'    Saving New Best Model With MSE Of: {str(mseForEpoch)}\n'
                torch.save(model.state_dict(),   modelPath)
            else:
                saveStr = ''

        mseAtEpochForVal.append(mseForEpoch)

        #TTR
        elapsed = time.time() - t
        print(f'TTR Epoch for {epoch+1} of {epochs}: {str(round(elapsed))} Seconds')
        print(f'Average Batch Time: {round(sum(batchTimes)/len(batchTimes),2)} Seconds')
        print(saveStr)

    tEnd = time.time()
    elapsed = tEnd - tStart
    print("Total TTR: " + str(round(elapsed)) + ' Seconds')


def plotEpochVsLoss():
    plt.plot(epochArr, losses, label='Loss')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plotEpochVsMse():
    plt.plot(epochArr, mseAtEpochForVal, label='Val Set')
    plt.title('Epoch vs MSE of Val Sets')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()