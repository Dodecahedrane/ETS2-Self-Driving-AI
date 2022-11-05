import redis
import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

global redisAngle
global redisSpeed

def RedisConn():
    #connect to redis
    redisAngle = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    redisSpeed = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

    try:
        redisSpeed.ping()
        redisAngle.ping()
        return True 
    except:
        print('Cannot Connect to redis')
        return False




def WheelLabel():
    label = input('What Angle?  ')
    if label == 's':
        return 's'
    elif label == 'x':
        print('Exiting')
        sys.exit()
    elif label.isnumeric():
        if int(label) > 0 and int(label) < 180:
            return label
        else:
            print("Out of bounds, must be between 0 and 180. Press 's' to skip")
            return WheelLabel()
    else:
        print("Must be interger between 0 and 180, press 's' to skip")
        return WheelLabel()

def WheelLabeler(path):
    print('Loading Wheel Angle Labeler')
    
    exist = os.path.exists(path)
    if exist:
        print('Directory Exists')
    else:
        print('Directory Does Not Exists')
        sys.exit()
    
    directory = os.fsencode(path+'/Angle/')
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):            
            img = mpimg.imread(path + '/Angle/' + filename)
            imgplot = plt.imshow(img)
            plt.show()
            
            label = WheelLabel()
            if label != 's':
                if redisAngle.get(label) == None:
                    redisAngle.set(filename, label)
                    os.remove(path + '/Angle/' +  filename)
            elif label == 's':
                print('Skipped')

def SpeedLabel():
    label = input('What Speed?  ')
    if label == 's':
        return 's'
    elif label == 'x':
        print('Exiting')
        sys.exit()
    elif label.isnumeric():
        if int(label) > 0 and int(label) < 100:
            return label
        else:
            print("Out of bounds, must be between 0 and 100. Press 's' to skip")
            return SpeedLabel()
    else:
        print("Must be interger between 0 and 100, press 's' to skip")
        return SpeedLabel()

def SpeedLabeler(path):
    print('Loading Speed Labeler')
    
    exist = os.path.exists(path)
    if exist:
        print('Directory Exists')
    else:
        print('Directory Does Not Exists')
        sys.exit()
    
    directory = os.fsencode(path)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img = mpimg.imread(path + '/' + filename)
            imgplot = plt.imshow(img)
            plt.show()
            
            label = SpeedLabel()
            if label != 's':
                if redisSpeed.get(label) == None:
                    redisSpeed.set(filename, label)
                    os.remove(path + '/' + filename)
                print(label + ' MPH')
            elif label == 's':
                print('Skipped')

def run():
    i = input("Type 'w' for wheel, 's' for speed or 'x' for exit: ")
    
    if(i == 's'):
        SpeedLabeler('SpeedLabel')
        
    elif(i == 'w'):
        WheelLabeler('WheelAngleLabel')
        
    elif(i == 'x'):
        print('Exiting Labeler')
        sys.exit()
        
    else:
        print('Not a valid input')
        run()


if __name__ == "__main__":
    print('ETS 2 Speed and Wheel Angle Labeling Tool')
    print('')
    print('')

    #connect to redis
    redisAngle = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    redisSpeed = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    status = False

    try:
        redisSpeed.ping()
        redisAngle.ping()
        status = True 
    except:
        print('Cannot Connect to redis')
        status = False

    if(status):
        i = ''
        run()
    else:
        print('Could not connect to local redis instance. please try again')