# Python Test code for testing the Action Unit, 
#    Emotion and Lie Detection recognition functions


import os,cv2
from lie2 import action_unit_emotion_lie_predict_from_30_frames,extract_frames_and_predict,args
import time


# load 30 images from the folder
#  and store in matrix 
def load_from_folder(folder_dir):
    list_images=[] 
    index=1
    for images in os.listdir(folder_dir):
        # load only first 30 images in folder
        if ((images.endswith(".jpg") or images.endswith(".png")) and index<=args.len_cut):
            list_images.append(cv2.imread(os.path.join(folder_dir,images)))
            index+=1
    return list_images

# Test routine for testing 1 sec of video as 30 images
# Set folder containing 30 images and load it for processing
folder_dir = "./folder_dir/"
#test_input_images=load_from_folder(folder_dir)

# Predict ML output 
#singleoutput= action_unit_emotion_lie_predict_from_30_frames(test_input_images)
#print(singleoutput)
# ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [-0.3012963831424713, -1.2281606197357178, -0.31528961658477783, -0.40303465723991394, 2.1766583919525146, -0.1118151992559433, -1.2411433458328247], 1)


start_time = time.time()  # Get the current time in seconds
# Test routine for testing a whole video from file
timestamped_outputs = extract_frames_and_predict(folder_dir+"t76.mp4")
# list of predictions for each second 
end_time = time.time()  # Get the current time again
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
#  Note to ignore exceptions
     
print(list(enumerate(timestamped_outputs,1)))
#print(timestamped_outputs)



