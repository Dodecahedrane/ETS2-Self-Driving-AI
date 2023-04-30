# Self Driving Agent for Euro Truck Simulator 2
## COMP 3000 Final Year Project


## Overview


This project aims to build an end to end lane keeping system that can be demonstrated in the game Euro Truck Driving Simulator 2, with the goal of useful lane keeping on the in-game highways.


## Architecture


There are two networks, the first estimates the steering angle of the truck. The second looks at the road to estimate the steering angle the truck should be driving at.


## Prerequisites


### Hardware


This project was tested on a system with a 5950x with 128GB of RAM and an RTX3090ti. Modifications to the training script may be needed if you have less VRAM or less system RAM.


1080p monitor (the cropping is based around a 1080p game resolution)


### Software


This project was built on Ubuntu 22.04 LTS, 20.04LTS or 18.04LTS should work but are untested.


A copy of ETS2 is required


Python - 3.10
PyTorch Version -
CUDA -


Python Libraries -


- OpenCV
- PIL
- MSS
- Pynput


OBS and Blender were used to record and edit the training footage.


## Getting Started


Firstly, clone this project:


`git clone https://github.com/Dodecahedrane/ETS2-Self-Driving-AI `


Secondly make sure you have all the prerequisites installed


Thirdly, choose your truck. It must be the same for both the training and inference to work (it looks at the steering wheel logo to determine steering angle). I have found a Volvo truck with the light cream interior to work the best. It is not tested with any others so anything else might require extra work.


## Training


### Training Data


You will need two types of training data for this project.


Firstly a video of the truck being driven correctly within the lane lines - this will be used to train the main driving network.


Secondly, a number of frames - around 50 - of the steering wheel in various lighting conditions in game. The wheel must be at 0 degrees pointing straight forward in all images. This will be used to train the network that estimates the current steering angle.


Then open the [pipeline.ipynb](https://github.com/Dodecahedrane/ETS2-Self-Driving-AI/blob/main/src/pipeline.ipynb) file to start the training.


There are five sections to this file.


The first section can extract frames from a video, just populatr the array with the file paths (can be Relative or Absolute) of the video you want to extract the frames from. This is designed to extract the frames from your edited training video.


The next section trains the steering network. You first need to gather approximately 50 images of the wheel at 0 degrees (straight forward) and place them in a folder called `StartingSteeringFrames`. You can then run the `AugmentSteeringData` method to create a training set for the steering network.


Once this training set is created the `trainSteering.train()` method will train the steering network. There are also two graphs showing loss and MSE rates at each epoch.


The next section augments and labels the driving data. To do this make sure all the driving frames you want to use are in a folder called `StartingDrivingFrames`. It will place the labeled and augmented images in a folder called `LabledDrivingFrames`.


You can then train the driving network with the `trainDriving.train()` method.


Once the above steps have been completed you'll have two models in the models folder that `run.py` can use to drive the truck in game.


## run.py


`run.py` is the main script that drives the truck. It will get the trained models from the `Models` folder.


To run it type `python3 src/run.py 0.05 60 5` in the command line when in the root of the repository.


There are three command line variables that alter the script. They are in the following order:


- Refresh Rate: How quickly the control loop runs
   - 0.025 to 0.075 is recommended
- Stopping Time: How long the control loop runs for
   - Because of the keyboard emulation it can be difficult to close the program while it is running. This stopped the program after a set amount of time (second)
- Start In: How long until the control loop starts
   - This waits a set amount of time before the control loop starts. This allows you to swap from the command line to the game window. You must have this window be active for it to work (ie, if you can steer with the keyboard, the control loop will not be able to steer either)


Within the run.py file there is one important variable that may need changing.


`bounding_box = {'top': 340, 'left': 1490, 'width': 1920, 'height': 1080}`


This tells the program where the ETS2 game window is on screen. This will need to be changed depending on your screen configuration.


The `top` and `left` variables are the xy pixel locations of the top left corner. With the size of the captured frame being 1080p (this is required, changing this would require a significant refactor the codebase)