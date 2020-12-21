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

class YoutubeVOS(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=2, increment=1, samples_per_video=12):
        data_dir = os.path.join(ROOT, 'Youtube-VOS')

        split = 'train' if train else 'valid'
        fullfolder = split + '_all_frames'

        blacklist = dict()

        self.root = data_dir
        self.imgdir = os.path.join(data_dir, split, 'JPEGImages')
        self.annodir = os.path.join(data_dir, split, 'Annotations')

        # with open(os.path.join(data_dir, split, 'meta_code.pkl'), 'rb') as f:
        #     meta = pickle.load(f)
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
        # vid = '65e0640a2a'
        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        # frames = [name[:5] for name in os.listdir(annofolder) if name not in self.blacklist[vid]]
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

                    frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
                    mask = [np.array(Image.open(os.path.join(annofolder, name+'.png'))) for name in sample_frame]

                    num_obj = int(mask[0].max())
                    sample_mask = sample_frame
                else:
                    sample_frame = frames
                    sample_mask = [name[:5] for name in os.listdir(annofolder)]
                    sample_mask.sort()

                    first_ref = sample_mask[0]
                    # clear ahead mask
                    sample_frame = [sample for sample in sample_frame if int(sample) >= int(first_ref)]
                    nframes = len(sample_frame)
                    frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
                    mask = [np.array(Image.open(os.path.join(annofolder, name+'.png'))) for name in sample_mask]

                    num_obj = max([int(msk.max()) for msk in mask])

                # clear dirty data
                for msk in mask:
                    msk[msk==255] = 0

            except OSError as ose:
                print(ose)
                num_obj = 0
                continue

        if self.train:
            num_obj = min(num_obj, MAX_TRAINING_OBJ)

        info = {'name': vid}
        info['frame'] = {
            'imgs': sample_frame,
            'masks': sample_mask
        }

        if not self.train:
            assert len(info['frame']['masks']) == len(mask), 'unmatched info-mask pair: {:d} vs {:d} at video {}'.format(len(info['frame']), len(mask), vid)

            num_ref_mask = len(mask)
            mask += [mask[0]] * (nframes - num_ref_mask)

        info['frame']['imgs'].sort()
        info['frame']['masks'].sort()
        info['palette'] = Image.open(os.path.join(annofolder, sample_frame[0]+'.png')).getpalette()
        info['size'] = frame[0].shape[:2]
        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        frame, mask = self.transform(frame, mask)
        # try:
        #     frame, mask = self.transform(frame, mask, False)
        # except Exception as exp:
        #     print(exp)
        #     print('Interruption at samples:')
        #     print(vid, sample_frame)
        #     exit()

        return frame, mask, num_obj, info

    def __len__(self):
        
        return self.length

class DaliYoutubeVOS(YoutubeVOS):

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        frames = [name[:5] for name in os.listdir(annofolder)]
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

        # print([cv2.imdecode(n, cv2.IMREAD_COLOR).shape for n in frame])
        # print([n.shape for n in mask])
        return frame, mask, [np.array([num_obj])] * nsamples, [np.array([self.max_obj])] * nsamples

register_data('VOS', YoutubeVOS)
register_data('DALIVOS', DaliYoutubeVOS)