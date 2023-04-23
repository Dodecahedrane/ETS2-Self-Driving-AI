import cv2
import os
import shutil

def setUpPaths():
    path = "Frames"
    exist = os.path.exists(path)
    if not exist:
        os.makedirs(path)
        print("Directoty Created")
    else:
        print('Directory Exists, deleted and recreated')
        #Uncomment these two lines if you want to delete the frames folder before adding more frames
        shutil.rmtree(path)
        os.makedirs(path)

def exportFrames(video):
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    
    videoNoExtension = video[:-4]
    
    while success:
        if count%10 == 0:
            cv2.imwrite('Frames/' + videoNoExtension + '_' + 'frame' + str(count) + '.jpg', image)

            success,image = vidcap.read()
            if not success:
                print('Error reading frame, aborting at frame number %d',count)
                break
            
        count += 1

    print('Finished Extracting Frames From ' + video)

def runExport(videoArr):
    setUpPaths()
    if len(videoArr) > 0:
        for v in videoArr:
            exportFrames(v)
    else:
       raise Exception("VideoArr is empty") 