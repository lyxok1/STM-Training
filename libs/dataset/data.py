import torch
import os
import math
import cv2
import numpy as np
import torch

import json
import yaml
import random
import pickle

from PIL import Image
from torch.utils.data import Dataset

DATA_CONTAINER = {}
ROOT = '/media/shh/966E77776E774F4D'
MAX_TRAINING_OBJ = 6
MAX_TRAINING_SKIP = 100

def multibatch_collate_fn(batch):

    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])

    objs = [torch.LongTensor([sample[2]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[3] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, objs, info

def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    oh = np.stack(oh, axis=2)

    return oh

def convert_one_hot(oh, max_obj):

    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k

    return mask

class BaseData(Dataset):

    def increase_max_skip(self):
        pass

    def set_max_skip(self):
        pass

class YoutubeVOS(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=2, increment=1, samples_per_video=12):
        data_dir = os.path.join(ROOT, 'Youtube-VOS')

        split = 'train' if train else 'valid'

        self.root = data_dir
        self.imgdir = os.path.join(data_dir, split, 'JPEGImages')
        self.annodir = os.path.join(data_dir, split, 'Annotations')

        with open(os.path.join(data_dir, split, 'meta.json'), 'r') as f:
            meta = json.load(f)

        self.info = meta['videos']
        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.videos = list(self.info.keys())
        self.length = len(self.videos) * samples_per_video
        self.max_obj = 12

        self.transform = transform
        self.train = train
        self.max_skip = max_skip
        self.increment = increment

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        # frames = [name[:5] for name in os.listdir(annofolder) if name not in self.blacklist[vid]]
        frames = [name[:5] for name in os.listdir(annofolder)]
        frames.sort()
        nframes = len(frames)

        num_obj = 0
        while num_obj == 0:
            # outer-loop check the transformed mask is valid
            while num_obj == 0:
                # inner-loop ensure the read mask is valid
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

                num_obj = int(mask[0].max())

            mask = [convert_mask(msk, self.max_obj) for msk in mask]

            info = {'name': vid}
            info['frame'] = [int(val['frames'][0][:5]) // 5 for idx, val in self.info[vid]['objects'].items()]
            info['frame'].sort()
            info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
            info['size'] = frame[0].shape[:2]

            if self.transform is None:
                raise RuntimeError('Lack of proper transformation')

            frame, mask = self.transform(frame, mask, False)

            if self.train:
                num_obj = 0
                for i in range(1, MAX_TRAINING_OBJ+1):
                    if torch.sum(mask[0, i]) > 0:
                        num_obj += 1
                    else:
                        break

        return frame, mask, num_obj, info

    def __len__(self):
        
        return self.length

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

        frame, mask = self.transform(frame, mask, False)

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

        frames = [name[:5] for name in os.listdir(annofolder)]
        frames.sort()
        nframes = len(frames)

        num_obj = 0
        while num_obj == 0:

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

        # if self.train:
        #     num_obj = min(num_obj, MAX_TRAINING_OBJ)

        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annofolder, frames[0]+'.png')).getpalette()
        info['size'] = frame[0].shape[:2]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        frame, mask = self.transform(frame, mask, False)

        if self.train:
            num_obj = 0
            for i in range(1, MAX_TRAINING_OBJ+1):
                if torch.sum(mask[0, i]) > 0:
                    num_obj += 1
                else:
                    break

        return frame, mask, num_obj, info

    def __len__(self):

        return self.length

DATA_CONTAINER['VOS'] = YoutubeVOS
DATA_CONTAINER['DAVIS16'] = Davis16
DATA_CONTAINER['DAVIS17'] = Davis17
