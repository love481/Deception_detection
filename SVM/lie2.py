# INSTRUCTIONS
# From come880412/Deception_detection repo
# Remove file: lie_GUI.py
# And use this file instead

# This file provides the following functions:
# 1. action_unit_emotion_lie_predict_from_30_frames(list_images):
#     input: list of 30 raw RGB images as list of matrix 480x640x3 
#     outputs: None or AUs  (12 Ekman Action Units as multi one hot with each either 0 or 1)
#              None or Emotions (7 emotions as real valued in range -3 to 8)
#              None or Lie  (single real number between 0 to 1
#                      with 1 for Lie and 0 for Truth)
#      Exception:
#         When no face is detected in at least 1 frame
#         output: None, None, None
#
#  Coding of AUs: Each index is either 1 or 0
#     
#     Index 1 --> AU1 = inner brow raiser
#     Index 2 --> AU2 = outer brow raiser
#     Index 3 --> AU4 = brow lower
#     Index 4 --> AU5 = upper lid raiser
#     Index 5 --> AU6 = cheek raiser 
#     Index 6 --> AU9 = nose raiser
#     Index 7 --> AU12 = lip corner puller
#     Index 8 --> AU15 = lip corner depressor
#     Index 9 --> AU17 = chin raiser
#     Index 10--> AU20 = lip Stretcher
#     Index 11--> AU25 = lips part
#     Index 12--> AU26 = jaw drop
#  Coding of Emotions: Each index is a real valued number from -3 to upto 8
#     Index 1 --> Happy
#     Index 2 --> Angry
#     Index 3 --> Disgust
#     Index 4 --> Fears
#     Index 5 --> Sad 
#     Index 6 --> Neutral
#     Index 7 --> Surprised
           
#        
# 2. extract_frames_and_predict
#      input: a path to a video file with exactly 30 frames per sec
#      output: a list of (AUs, Emotions, Lie) as above 
#                   indexed by 1 sec in the video
#                    so that:
#                         1st item in the list corresponds to 1st sec in video 
#                         2nd item in the list corresponds to 2nd sec in video etc


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
#Feature extraction
import model.Emotion.lie_emotion_process as emotion
import model.action_v4_L12_BCE_MLSM.lie_action_process as action
from model.action_v4_L12_BCE_MLSM.config import Config
from joblib import load
import time
parser = argparse.ArgumentParser()

# 30 frames for combined predictions
parser.add_argument('--len_cut', default=30, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d', '3d'])
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',help= '0 is self-attention; 1 is self + relation-attention')
parser.add_argument('--preTrain_path', '-pret', default='./model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar', type=str, help='pre-training model path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num


# The input 30 frames (corresponding to 1 sec of video) is processed by the following steps:
# 1. Face detection
# 2. Face alignment
# 3. Feature extraction
# 4. Action unit prediction
# 5. Emotion unit prediction
# 6. Deception prediction

#load model
Retina = retina_face(crop_size = 224, args = args) # Face detection
Emotion_class = emotion.Emotion_FAN(args = args)   # Emotion classificaton
Action_class = action.Action_Resnet(args= Config())# Action Multi label classification
SVM_model = load('./model/SVM_model/se_res50+EU/split_svc_acc0.720_AUC0.828.joblib') ##SVM model

# Class for Action units prediction
class AU_pred():
    def  __init__ (self,image):
        super(AU_pred ,self). __init__ ()
        self.face = image
    def run(self):
        # Action unit labels and Embedding vector of size (1,2048)
        logps, emb = Action_class._pred(self.face,Config)
        return logps.tolist(), emb.tolist()

# Main Class used for emotion units and deception prediction
class show():
    def  __init__ (self, frame_list ,frame_AU,log):
        super(show,self). __init__ ()
        self.frame_embed_list = frame_list 
        self.frame_emb_AU = frame_AU 
        self.log = log

    def set_inputs(self,frame_list ,frame_AU,log):
        self.frame_embed_list = frame_list 
        self.frame_emb_AU = frame_AU 
        self.log = log

    def pred(self):
        AU_list = self.log.tolist()[0]
        for index,i in enumerate(AU_list):
            # threshold for action units
            if i >= 0.01:
                AU_list[index] = 1
            else:
                AU_list[index] = 0
        # Emotion unit labels,_, relation_embedding of size (1,1024)
        pred_score, _, relation_embedding = Emotion_class.validate(self.frame_embed_list)
        # Concatenated feature of size (1,3072)
        feature = np.concatenate((self.frame_emb_AU,relation_embedding.cpu().numpy()), axis = 1)
        # Lie_pred using SVM model
        results = SVM_model.predict(feature) 
        # Action unit labels, Emotion unit labels and Lie prediction
        return AU_list, pred_score, results
    def run(self):
        logps,  pred_score, results  = self.pred()
        return logps, pred_score.tolist()[0], int(results[0])


# Class for predicting lie. Uses AU prediction
class lie():
    def __init__(self):
        super(lie, self).__init__()
        # stores the input raw frames
        self.frame_embed_list = [] 
        self.frame_emb_AU = []
        self.log = [] 
        self.userface =[] 
        self.index = 0
        # Instantiate object of show class
        self.show_obj=show(self.frame_embed_list, self.frame_emb_AU,self.log) 

    # stores the Action embedding of each frames and corresponding labels
    def AU_store(self,AU_emb,log):
        AU_emb = torch.FloatTensor(AU_emb)
        log = torch.FloatTensor(log)
        self.frame_emb_AU.append(AU_emb.cpu().numpy())
        self.log.append(log.cpu().numpy())
 
    # empty previous inputs
    def empty_attribute(self):
        self.frame_embed_list = []
        self.frame_emb_AU = [] 
        self.log = [] 
        self.userface =[] 
        self.index = 0

    
    def predictions(self,list_images):
        # list_images: list of 30 RGB image of matrix having pixel values ranges from (0-255)
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
                    self.frame_embed_list.append(out_raw)
                    lnd_AU = AU_pred(out_raw) 
                    logps, emb=lnd_AU.run()
                    self.AU_store(emb,logps) 
        
        # check whether 30 frames are stored or not, it is limited when no person is detected by face detector
        ## Else return None
        if len(self.frame_embed_list) == args.len_cut:
            self.frame_emb_AU = np.array(self.frame_emb_AU)
            # Mean of all the embeddings of 30 image matrix for action units embeddings
            self.frame_emb_AU = np.mean(self.frame_emb_AU, axis = 0)
            self.log = np.array(self.log)
            # Mean of all action units labels of 30 image matrix
            self.log = np.mean(self.log, axis = 0)
            self.show_obj.set_inputs(self.frame_embed_list, self.frame_emb_AU,self.log)
            logps, pred_score, results=self.show_obj.run() 
            # return the emotion labels, action labels and lie predictions
            return logps, pred_score, results 
        else:
            return None,None,None


              
# 2. extract_frames_and_predict
#      input: a path to a video file with exactly 30 frames per sec
#      output: a list of (AUs, Emotions, Lie) as above 
#                   indexed by 1 sec in the video
#                    so that:
#                         1st item in the list corresponds to 1st sec in video 
#                         2nd item in the list corresponds to 2nd sec in video etc
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
        # Ignore the last few frames if its number is less than 30
        if index>total_pred*args.len_cut or Ret == False:
            return list_prediction
        list_images.append(Mat)  
        # predict the labels for 30 frames
        if index%args.len_cut == 0:
            pred=api.predictions(list_images)
            list_prediction.append(pred)
            #print(list(enumerate(list_prediction,1)))
            api.empty_attribute()
            list_images=[]
        index+=1
    cap.release()   
    del api
    return list_prediction


# 1. action_unit_emotion_lie_predict_from_30_frames(list_images):
#     input: list of 30 raw RGB images as list of matrix 480x640x3 
#     outputs: None or AUs  (12 Ekman Action Units as multi one hot with each either 0 or 1)
#              None or Emotions (7 emotions as real valued)
#              None or Lie  (single real number between 0 to 1
#                      with 1 for Lie and 0 for Truth)
#      Exception:
#         When no face is detected in at least 1 frame
#         output: None, None, None
def action_unit_emotion_lie_predict_from_30_frames(list_images):
    api=lie()
    return api.predictions(list_images)


