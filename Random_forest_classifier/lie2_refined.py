# INSTRUCTIONS
# The Action unit model is taken from lingjivoo/OpenGraphAU
# Random Forest classifier model is designed from the real-life-trail-datasets 
# using the same model given above

# This file provides the following functions:
# arguments n="len_cut" (number of frames) is the hyperparameter which can be tuned for action units label predictions
# e.g. setting len_cut=60 means, we output deception label and corresponding AUs each 2 second of video,
# simply divide by 30 to get the outputs at required timeseconds.
# 1. action_unit_lie_predict_from_n_frames(list_images):
#     input: list of n raw RGB images as list of matrix 480x640x3 where n is multiple of 30(fps)
#     outputs: None or AUs  (41 Ekman Action Units both unilateral and bilateral as multi class labeling with each value either 0 or 1)
#              None or Lie  (single real number between 0 to 1
#                      with 1 for Lie and 0 for Truth)
#      Exception:
#         When no face is detected in at least 1 frame
#         output: None, None
#
#  Coding of AUs: Each index is either 1 or 0
#     
#     Index 1 --> AU1 = inner brow raiser
#     Index 2 --> AU2 = outer brow raiser
#     Index 3 --> AU4 = brow lower
#     Index 4 --> AU5 = upper lid raiser
#     Index 5 --> AU6 = cheek raiser 
#     Index 6 --> AU7 = lid tightener
#     Index 7 --> AU9 = nose wrinkle
#     Index 8 --> AU10 = upper lip raiser
#     Index 9 --> AU11 = nasolabial deepener
#     Index 10 --> AU12 = lip corner puller
#     Index 11 --> AU13 = sharp lip puller
#     Index 12 --> AU14 = dimpler
#     Index 13 --> AU15 = lip corner depressor
#     Index 14 --> AU16 = lower lip depressor
#     Index 15 --> AU17 = chin raiser
#     Index 16 --> AU18 = lip pucker
#     Index 17 --> AU19 = tongue show
#     Index 18--> AU20 = lip Stretcher
#     Index 19--> AU22 = lip funneler
#     Index 20--> AU23 = lip tightener
#     Index 21--> AU24 = lip pressor
#     Index 22--> AU25 = lips part
#     Index 23--> AU26 = jaw drop
#     Index 24--> AU27 = mouth stretch
#     Index 25--> AU32 = lip bite
#     Index 26--> AU38 = nostril dilator
#     Index 27--> AU39 = nostril compressor
#     Index 28--> AUL1 = left inner brow raiser
#     Index 29--> AUR1 = right inner brow raiser
#     Index 30--> AUL2 = left outer brow raiser
#     Index 31--> AUR2 = right outer brow raiser
#     Index 32--> AUL4 = left brow lowerer
#     Index 33--> AUR4 = right brow lowerer
#     Index 34--> AUL6 = left cheek raiser
#     Index 35--> AUR6 = right cheek raiser
#     Index 36--> AUL10 = left upper lip raiser
#     Index 37--> AUR10 = right upper lip raiser
#     Index 38--> AUL12 = left nasolabial deepener
#     Index 39--> AUR12 = right nasolabial deepener
#     Index 40--> AUL14 = left dimpler
#     Index 41--> AUR14 = right dimpler
           
#        
# 2. extract_frames_and_predict
#      input: a path to a video file with exactly 30 frames per sec
#      output: a list of (AUs, Lie) as above 
#                   indexed by each 2 sec in the video
#                    so that:
#                         1st item in the list corresponds to 2 sec in video 
#                         2nd item in the list corresponds to 4 sec in video etc


from model.ANFL import MEFARG
from utils import *
from conf import get_config,set_env
import pandas as pd
import cv2
import numpy as np
import os
import argparse
from random import sample
#torch
import torch
import skimage
from skimage import img_as_ubyte
from model.Facedetection.config import device
#face detection
from model.Facedetection.utils import align_face
from model.Facedetection.RetinaFace.RetinaFaceDetection import retina_face
from joblib import load

parser = argparse.ArgumentParser()

# 60 frames for combined predictions which can be tuned. It is the hyperparameter.
parser.add_argument('--len_cut', default=60, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

# Action units
AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
           '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
# AU_ids=['AU_'+ x for x in AU_ids]
# print(AU_ids)

# The input n frames as defined in arguments len_cut is processed by the following steps:
# 1. Face detection
# 2. Face alignment
# 4. Action unit prediction

## define configuration
conf = get_config()
conf.evaluate = True
set_env(conf)
# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Retina = retina_face(crop_size = 224, args = args) # Face detection
# action unit recognition model
Action_class = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
net = load_state_dict(Action_class, conf.resume).to(device)
net.eval()

# Class for Action units prediction
class AU_pred():
    def  __init__ (self,image):
        super(AU_pred ,self). __init__ ()
        self.img_transform = image_eval()
        self.face = image
    def run(self):
        img=Image.fromarray(self.face)
        img_ = self.img_transform(img).unsqueeze(0)
        img_ = img_.cuda()
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()
        return pred

# Main Class used for action units and deception prediction
class show():
    def  __init__ (self,log,w_s):
        super(show,self). __init__ ()
        self.log = log
        self.window_size=w_s
        self.sequence_length=0
        # random forest classifier for deception detection based on the action units
        self.dtree=load('random_forest_classifier_deception.joblib')
        self.x=pd.DataFrame()

    def set_inputs(self,log):
        self.log = log
        self.sequence_length=len(self.log)
        self.x=pd.DataFrame()

    def pred(self):
        AU_list = self.log
        # Compute sum of first window of size self.window_size
        window_sum = sum(AU_list[:self.window_size])
        max_sum = window_sum
        # Action unit labels
        max_sum=(window_sum>(self.window_size*0.70)).astype(int)
        self.x[AU_ids]=[list(max_sum)]
        return list(max_sum),self.dtree.predict(self.x)[0]
    def run(self):
        AU_list,deceptive_label  = self.pred()
        if deceptive_label=='truthful':
            deceptive_label=0
        else:
            deceptive_label=1

        return AU_list,deceptive_label


# Class for predicting lie. Uses AU prediction
class lie():
    def __init__(self):
        super(lie, self).__init__()
        self.log = [] 
        self.userface =[] 
        self.index = 0
        # Instantiate object of show class
        self.show_obj=show(self.log,args.len_cut) 

    # stores the action units labels
    def AU_store(self,log):
        self.log.append(log)
 
    # empty previous inputs
    def empty_attribute(self):
        self.log = [] 
        self.userface =[] 
        self.index = 0

    
    def predictions(self,list_images):
        # list_images: list of n RGB image of matrix having pixel values ranges from (0-255)
        for i,img in enumerate(list_images):
            if i<args.len_cut:
                im = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                image = skimage.img_as_float(im).astype(np.float32)
                frame = img_as_ubyte(image)
                # face detection using retina face from the raw RGB images of matrix size 480x640x3 having pixel values ranges between (0-255)
                _, output_raw, output_points,bbox,face_list = Retina.detect_face(frame) 
                if len(bbox) == 1:
                    self.index = 0
                    self.userface = face_list[self.index]
                elif len(bbox) >= 2:
                    if len(self.userface):
                        dist_list = []
                        face_list = np.array(face_list)
                        for i in range(len(bbox)):
                            dist = np.sqrt(np.sum(np.square(np.subtract(self.userface[:], face_list[i, :]))))
                            dist_list.append(dist)
                        dist_list = np.array(dist_list)
                        self.index = np.argmin(dist_list)

                if(len(output_points)):
                    # face_align
                    out_raw = align_face(output_raw, output_points[self.index], crop_size_h = 112, crop_size_w = 112)
                    out_raw = cv2.resize(out_raw,(224, 224))
                    # out_raw: face aligned image of matrix 224x224x3 ranges between (0-255)
                    lnd_AU = AU_pred(out_raw)
                    logps=lnd_AU.run()
                    logps=np.array(logps>0.5).astype(int)
                    self.AU_store(logps) 
        if len(self.log) == args.len_cut:
            self.show_obj.set_inputs(self.log)
            logps,deceptive_label=self.show_obj.run() 
            return logps,deceptive_label
        else:
            return None,None

def extract_frames_and_predict(folder_dir_video):
    cap = cv2.VideoCapture(folder_dir_video)
    index = 1 
    list_prediction=[] 
    list_images=[]    
    # frame_total: total number of frames in videos
    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # total_pred: total number of predictions in video
    total_pred=int(frame_total/args.len_cut)
    # object to process the frames and predict the labels
    api=lie() 
    while cap.isOpened():
        # Ret: boolean value to check whether the frame is present or not
        # Mat: frame in matrix format sampled from video
        Ret, Mat = cap.read()
        # index: index of the frame
        # Ignore the last few frames if its number is less than n frames
        if index>total_pred*args.len_cut or Ret == False:
            return list_prediction
        list_images.append(Mat)  
        # predict the labels for n frames
        if index%args.len_cut == 0:
            pred=api.predictions(list_images)
            list_prediction.append(pred)
            api.empty_attribute()
            list_images=[]
        index+=1
    cap.release()   
    del api
    return list_prediction

def action_unit_lie_predict_from_n_frames(list_images):
    api=lie()
    return api.predictions(list_images)



