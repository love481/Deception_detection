# Python Test code for testing the Action Unit, 
#    Emotion and Lie Detection recognition functions


import os,cv2
from lie2 import extract_frames_and_predict,args
import time
import pandas as pd
import numpy as np
# load 30 images from the folder
#  and store in matrix 
list_videos_names=[] 
def load_from_folder(folder_dir):
    for videos in os.listdir(folder_dir):
        if (videos.endswith(".mp4")):
            list_videos_names.append(os.path.join(folder_dir,videos))

# Test routine for testing 1 sec of video as 30 images
# Set folder containing 30 images and load it for processing
folder_dir_deceptive = "./MSPLAB_YouTube/Deceptive"
folder_dir_truthful = "./MSPLAB_YouTube/Truthful"
columns = ['id', '1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
           '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14', 'class']
load_from_folder(folder_dir_deceptive)
load_from_folder(folder_dir_truthful)
list_videos_names.sort()
print(list_videos_names)
#Predict ML output 
start_time = time.time()  # Get the current time in seconds

data=pd.DataFrame()
action_unit_data=[]
# Test routine for testing a whole video from file
for i in list_videos_names:
    values=extract_frames_and_predict(i)
    list_names=i.split('/')
    values.insert(0, list_names[3])
    values.append(list_names[2].lower())
    action_unit_data.append(values)
data[columns]=action_unit_data
print(data)
data.to_csv('msplab_data.csv')
# list of predictions for each second 
end_time = time.time()  # Get the current time again
execution_time = end_time - start_time


print(f"Execution time: {execution_time:.6f} seconds")
#  Note to ignore exceptions
     
print(action_unit_data)



