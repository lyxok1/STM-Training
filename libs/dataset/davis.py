import torch
import os
import math
import cv2
import numpy as np

import json
import yaml
import random
import lmdb
import pickle

from PIL import Image
from ..utils.logger import getLogger

from .data import *

class Davis16(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=5, increment=5, samples_per_video=12):
        
        data_dir = os.path.join(ROOT, 'DAVIS16')
        dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_dir, 'JPEGImages', '480p')
        self.annodir = os.path.join(data_dir, 'Annotations', '480p')

        self.root = data_dir

        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']

            targetset = 'train' if train else 'val'
            self.info = db
            self.videos = [info['name'] for info in db if info['set']==targetset]

        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.length = samples_per_video * len(self.videos)
        self.max_skip = max_skip
        self.max_obj = 1
        self.increment = increment
        
        self.transform = transform
        self.train = train

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        frames = [name[:5] for name in os.listdir(annofolder)]
        frames.sort()
        nframes = len(frames)

        if self.train:
            last_sample = -1
            sample_frame = []
            nsamples = min(self.sampled_frames, nframes)
            for i in range(nsamples):
                if i == 0:
                    last_sample = random.sample(range(0, nframes-nsamples+1), 1)[0]
                else:
                    last_sample = random.sample(
                        range(last_sample+1, min(last_sample+self.max_skip+1, nframes-nsamples+i+1)), 
                    1)[0]
                sample_frame.append(frames[last_sample])
        else:
            sample_frame = frames

        frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
        mask = [np.array(Image.open(os.path.join(annofolder, name+'.png'))) for name in sample_frame]
        num_obj = max([int(msk.max()) for msk in mask])
        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
        info['size'] = frame[0].shape[:2]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        try:
            frame, mask = self.transform(frame, mask)
        except Exception as exp:
            print(exp)
            print('Interruption at samples:')
            print(vid, sample_frame)
            exit()

        return frame, mask, num_obj, info

    def __len__(self):
        return self.length

class Davis17(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=5, increment=5, samples_per_video=12):
        
        data_dir = os.path.join(ROOT, 'DAVIS17')

        dbfile = os.path.join(data_dir, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_dir, 'JPEGImages', '480p')
        self.annodir = os.path.join(data_dir, 'Annotations', '480p')

        self.root = data_dir
        self.max_obj = 0

        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']

            targetset = 'train' if train else 'val'
            # targetset = 'training'
            self.info = db
            self.videos = [info['name'] for info in db if info['set']==targetset]

            for vid in self.videos:
                objn = np.array(Image.open(os.path.join(self.annodir, vid, '00000.png'))).max()
                self.max_obj = max(objn, self.max_obj)

        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.length = samples_per_video * len(self.videos)
        self.max_skip = max_skip
        self.increment = increment
        
        self.transform = transform
        self.train = train

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        frames = [name[:5] for name in os.listdir(imgfolder)]
        frames.sort()
        nframes = len(frames)

        num_obj = 0
        while num_obj == 0:

            try:
                if self.train:
                    last_sample = -1
                    sample_frame = []

                    nsamples = min(self.sampled_frames, nframes)
                    for i in range(nsamples):
                        if i == 0:
                            last_sample = random.sample(range(0, nframes-nsamples+1), 1)[0]
                        else:
                            last_sample = random.sample(
                                range(last_sample+1, min(last_sample+self.max_skip+1, nframes-nsamples+i+1)), 
                            1)[0]
                        sample_frame.append(frames[last_sample])
                else:
                    sample_frame = frames

                frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
                mask = [np.array(Image.open(os.path.join(annofolder, name+'.png'))) for name in sample_frame]
                # clear dirty data
                for msk in mask:
                    msk[msk==255] = 0

                num_obj = mask[0].max()

            except FileNotFoundError as fnfe:
                # placeholder
                print('[WARNING] build palce holder for video mask')
                mask = [np.array(Image.open(os.path.join(annofolder, '00000.png')))] * nframes

                # clear dirty data
                for msk in mask:
                    msk[msk==255] = 0

                num_obj = mask[0].max()

            except OSError as ose:
                num_obj = 0
                continue


        if self.train:
            num_obj = min(num_obj, MAX_TRAINING_OBJ)

        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
        info['size'] = frame[0].shape[:2]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        try:
            frame, mask = self.transform(frame, mask)
        except Exception as exp:
            print(exp)
            print('Interruption at samples:')
            print(vid, sample_frame)
            exit()

        return frame, mask, num_obj, info

    def __len__(self):

        return self.length

class DaliDavis17(Davis17):

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        frames = [name[:5] for name in os.listdir(imgfolder)]
        frames.sort()
        nframes = len(frames)

        last_sample = -1
        sample_frame = []

        nsamples = min(self.sampled_frames, nframes)
        for i in range(nsamples):
            if i == 0:
                last_sample = random.sample(range(0, nframes-nsamples+1), 1)[0]
            else:
                last_sample = random.sample(
                    range(last_sample+1, min(last_sample+self.max_skip+1, nframes-nsamples+i+1)), 
                1)[0]
            sample_frame.append(frames[last_sample])

        frame = [np.frombuffer(open(os.path.join(imgfolder, name+'.jpg'), 'rb').read(), np.uint8) 
                    for name in sample_frame]

        mask = [np.array(Image.open(os.path.join(annofolder, name+'.png')))[:, :, None].astype(np.float32) for name in sample_frame]
        # clear dirty data
        for msk in mask:
            msk[msk==255] = 0

        num_obj = mask[0].max()

        return frame, mask, [np.array([num_obj])] * nsamples, [np.array([self.max_obj])] * nsamples


register_data('DAVIS16', Davis16)
register_data('DAVIS17', Davis17)
register_data('DALIDAVIS17', DaliDavis17)