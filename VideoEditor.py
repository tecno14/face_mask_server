import os
import random
import uuid
import time
import numpy as np
from datetime import datetime
import math
# video edit
from moviepy.editor import ImageSequenceClip, TextClip, VideoFileClip,\
    concatenate_videoclips#, ImageClip, CompositeVideoClip
from moviepy.audio.AudioClip import AudioArrayClip
#face track
import cv2
#voice
import librosa
from AudioLib.AudioProcessing import AudioProcessing

#get facial classifiers
CCPATH = './Lib/site-packages/cv2/haar-cascade-files-master/'
face_cascade = cv2.CascadeClassifier(CCPATH +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(CCPATH +'haarcascade_eye.xml')

class VideoEditor:
    
    DEFAULT_MASK_PATH = os.path.abspath("./masks/1.png")
    TEMP_FOLDER_PATH = os.path.abspath("./temp_files")
    DEFAULT_FILENAME = "output.mp4"
    BLURE_FULL_IMG = (120, 120)
    BLURE_OLD_FACES = (75, 75)
    CURRENT_DIR = None

    def __init__(self, input_video, mask_path = None, output_video = None):
        
        # current dir reset
        if VideoEditor.CURRENT_DIR is None:
            VideoEditor.CURRENT_DIR = os.getcwd()
        else:
            os.chdir(VideoEditor.CURRENT_DIR)

        # check if input video exist
        if not os.path.isfile(input_video):
            raise Exception('Error : input video path not found')
        self.input_video = os.path.abspath(input_video)
        
        # make tmp folder
        if not os.path.isdir(VideoEditor.TEMP_FOLDER_PATH):
            os.mkdir(VideoEditor.TEMP_FOLDER_PATH)
        now = datetime.now()
        self.tmp_folder = os.path.abspath(os.path.join(VideoEditor.TEMP_FOLDER_PATH,\
            now.strftime("%Y_%m_%d_%H_%M_%S___") + str(uuid.uuid4()) ))
        print("temp folder '{}'".format(self.tmp_folder))
        os.mkdir(self.tmp_folder)

        # output video path
        if output_video is None:
            filename = VideoEditor.DEFAULT_FILENAME
            output_video = os.path.join(self.tmp_folder, filename)
        else:
            output_video = os.path.abspath(output_video)
        self.output_video = output_video
        
        # mask
        if mask_path is None:
            mask_path = VideoEditor.DEFAULT_MASK_PATH
        print("-Reading mask ...")
        if not os.path.isfile(mask_path):
            raise Exception('Error : mask not found')
        self.mask = cv2.imread(mask_path)
        self.original_witch_h, self.original_witch_w,_ = self.mask.shape
        self.mask_gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        _, self.original_mask = cv2.threshold(self.mask_gray, 10, 255, cv2.THRESH_BINARY_INV)
        self.original_mask_inv = cv2.bitwise_not(self.original_mask)

        # default blure area
        self.blure_x_max = self.blure_y_max = 0
        self.blure_x_min = self.blure_y_min = math.inf
        
        self.hide_full_frame = True
        self.last_faces = None

    def __del__(self):
        
        print("\n\n")
        # deleting temp folder if empty
        if not any(os.scandir(self.tmp_folder)):
            print('no files found !!')
            print('-deleting {}'.format(self.tmp_folder))
            os.rmdir(self.tmp_folder)

    def hide_face(self):
                
        # change current directory to temp folder
        os.chdir(self.tmp_folder)

        print("-Reading video ...")
        clip = VideoFileClip(self.input_video)

        print("-Edit video frames ...")
        new_frames = []
        count = 0
        count_process = 0
        clip_frames_count = clip.fps * clip.duration + 1
        proccess_at_fps = 5
        tmp = -1
        start_time = time.monotonic()
        for frame in clip.iter_frames():
            count += 1; tmp += 1
            
            if tmp == proccess_at_fps:
                tmp = 0
            
            process = False
            if tmp == 0 or self.last_faces is None:
                count_process += 1; process = True
                
            frame = self.hide_face_frame(frame, process)
            new_frames.append(frame)
            
            print("time: {} s - frames percentage: {} %    ".format(\
                round(time.monotonic() - start_time), \
                round(count / clip_frames_count * 100, 2)), end="\r")

        print("processed frames : ( {} / {} )             ".format(count_process,count))
        print("time taken : {} s ".format(round(time.monotonic() - start_time)))
        print("-Generate text ...")
        txt = TextClip("Hey you ! \n\n Join Fadfada \n Today", color='FloralWhite',
        size = clip.size, bg_color='DodgerBlue4',
        font = "Century-Schoolbook-Italic", fontsize= clip.w / 12)
        
        print("-concatinate ImageSequence ...")
        new_clip = ImageSequenceClip(new_frames, fps=clip.fps)

        # voice
        print("-voice processing ...")
        tasks_count = 4; current_task_id = 1
        print("Loading ( {} / {} )".format(current_task_id, tasks_count), end="\r"); current_task_id += 1
        fs = clip.audio.fps
        audio_arr = clip.audio.copy().to_soundarray()
        # to mono
        print("Loading ( {} / {} )".format(current_task_id, tasks_count), end="\r"); current_task_id += 1
        new_audio_arr = []
        for e in audio_arr:
            new_audio_arr.append((e[0] / 2) + (e[1] / 2))
        audio_arr = np.array(new_audio_arr)
        # effect
        print("Loading ( {} / {} )".format(current_task_id, tasks_count), end="\r"); current_task_id += 1
        y_shifted = librosa.effects.pitch_shift(audio_arr, fs, n_steps=random.randrange(-6, -3))
        sound = AudioProcessing(y_shifted, fs)
        # sound.set_bandpass(300, 3000)
        audio_arr = sound.get_data()
        # to dual
        print("Loading ( {} / {} )".format(current_task_id, tasks_count), end="\r"); current_task_id += 1
        new_audio_arr = []
        for e in audio_arr:
            new_audio_arr.append([e, e])
        audio_arr = np.array(new_audio_arr)
        audio = AudioArrayClip(audio_arr, fps=fs)

        print("-concatinate Video and Audio...")
        new_clip = concatenate_videoclips([new_clip, txt.set_duration(3)]).set_audio(audio)
        new_clip.write_videofile(self.output_video)
        # new_clip.write_videofile(self.output_video, bitrate="3000k")

        # reset current directory
        os.chdir(VideoEditor.CURRENT_DIR)

        return self.output_video

    def hide_face_frame(self, img, detect_faces = False):

        img = img.copy()

        # convert to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces in image using classifier
        faces = []
        if detect_faces:
            faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)  
        
        # if there no faces in current frame blur it
        if len(faces) == 0 and self.hide_full_frame:
            img = cv2.blur(img, VideoEditor.BLURE_FULL_IMG)
        else:
            self.hide_full_frame = False

        if self.blure_y_min != math.inf:
            img[self.blure_y_min:self.blure_y_max, self.blure_x_min:self.blure_x_max] =\
                cv2.blur(img[self.blure_y_min:self.blure_y_max, self.blure_x_min:self.blure_x_max],\
                    VideoEditor.BLURE_OLD_FACES)

        # put mask on each face
        for face in faces:
            img = self.put_mask_on_face(img, face)

        # last mask saved
        if len(faces) == 0 and self.last_faces is not None:
            for face in self.last_faces:
                img = self.put_mask_on_face(img, face)
        if len(faces) != 0:
            self.last_faces = faces

        return img

    def put_mask_on_face(self, img, face):

        #get shape of img
        img_h,img_w,_ = img.shape
        X, Y, W, H = face
        
        #retangle for testing purposes
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = W
        face_h = H
        face_x1 = X
        face_x2 = face_x1 + face_w
        face_y1 = Y
        face_y2 = face_y1 + face_h

        #witch size in relation to face by scaling
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * self.original_witch_h / self.original_witch_w)

        #setting location of coordinates of witch
        # witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        # witch_x2 = witch_x1 + witch_width
        # witch_y1 = face_y1 - int(face_h*1.25)
        # witch_y2 = witch_y1 + witch_height 

        witch_x1 = face_x1 - int(face_w*0.11)
        witch_x2 = face_x2 + int(face_w*0.11)
        witch_y1 = face_y1 - int(face_h*0.3) #int(witch_height/6)
        witch_y2 = face_y2 + int(face_h*0.3) #face_y2 + int(witch_height/6)

        #check to see if out of frame

        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h
        
        mask = self.mask.copy()

        #Account for any out of frame changes
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1
        
        #resize witch to fit on face
        mask = cv2.resize(mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        witch2 = cv2.resize(self.original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(self.original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

        #take ROI for witch from background that is equal to size of witch image
        roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

        #original image in background (bg) where witch is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = witch2)            
        roi_fg = cv2.bitwise_and(mask,mask,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        self.blure_x_min = min(self.blure_x_min, witch_x1)
        self.blure_x_max = max(self.blure_x_max, witch_x2)
        self.blure_y_min = min(self.blure_y_min, witch_y1)
        self.blure_y_max = max(self.blure_y_max, witch_y2)
        #print("{} , {} , {} , {}".format(self.blure_x_min,self.blure_x_max,self.blure_y_min,self.blure_y_max))

        #put back in original image
        img[witch_y1:witch_y2, witch_x1:witch_x2] = dst

        return img


# test section
# all_start_time = time.monotonic()
# x=1
# if x==1:
#     test_video = VideoEditor("./videos/2.mp4","./masks/6.png","./temp_files/2-6.mp4")
#     test_video.hide_face()
    
#     test_video = VideoEditor("./videos/1.mp4","./masks/6.png","./temp_files/1-6.mp4")
#     test_video.hide_face()
# else:
#     test_video = VideoEditor("./videos/2.mp4","./masks/5.png","./temp_files/last.mp4")
#     img_ = cv2.imread('./imgs/1.jpg')
#     img_ = test_video.hide_face_frame(img_)

#     cv2.imshow('img',img_)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# print("all time taken : {} s".format(round(time.monotonic() - all_start_time)))