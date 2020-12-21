import nvidia.dali as dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.pipeline as pipe

import os
import random
import numpy as np
import torch
import cv2

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from .data import convert_mask, MAX_TRAINING_OBJ

# debug
def vis(frames, masks, objs):

    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 128, 0), (0, 255, 128), (0, 128, 255),
        (128, 255, 0), (128, 0, 255), (255, 0, 128),
        (128, 128, 255), (128, 255, 128), (255, 128, 128)
    ]

    imgs = (frames.squeeze().cpu().numpy() * 0.226 + 0.45) * 255
    imgs = imgs.astype(np.uint8)
    for t in range(imgs.shape[0]):
        img = imgs[t][:, :, (2, 1, 0)]
        msk = masks[t].squeeze().cpu().numpy()
        num = min(len(palette), int(objs))
        trans_m = np.zeros(img.shape, dtype=np.uint8)

        for k in range(1, 1 + num):
            print(num, np.sum(msk == k))
            trans_m[msk==k, :] = palette[k-1]

        im = cv2.addWeighted(img, 0.5, trans_m, 0.5, 0.0)
        cv2.imwrite('test_{:>03d}_raw.jpg'.format(t), img)
        cv2.imwrite('test_{:>03d}.jpg'.format(t), im)

class DaliLoader(object):

    def __init__(self, datasets, freq):

        self.datasets = datasets
        self.freq = freq
        self.range = list(range(len(datasets)))

    def __iter__(self):

        for i in range(self.freq):
            random.shuffle(self.range)
            for idx in self.range:
                yield self.datasets[idx]

    def __len__(self):
        return len(self.datasets) * self.freq

class TrainPipeline(pipe.Pipeline):

    def __init__(self, reader, batch_size, size, device_id=0, num_threads=1):
        super(TrainPipeline, self).__init__(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id
            )

        self.rotate_range = [-10.0, 10.0]
        self.noise_range = [0, 5.0]
        self.contrast_range = [0.97, 1.03]
        self.crop_range = [0.9, 1.0]
        self.shear_range = [0.04, 0.025]
        # self.mean = ops.Constant(fdata=[0.485, 0.456, 0.406])
        # self.std = ops.Constant(fdata=[0.229, 0.224, 0.225])
        self.size = size

        # source
        self.input = ops.ExternalSource(source=reader, num_outputs=4)
        self.affine_source = ops.ExternalSource()
        self.mirror_rng = ops.ExternalSource()

        # transform
        self.decode = ops.ImageDecoder(output_type=types.RGB)
        self.rotate_rng = ops.Uniform(range=self.rotate_range)
        # self.noise_rng = ops.Uniform(range=self.noise_range)
        # self.contrast_rng = ops.Uniform(range=self.contrast_range)
        # self.mirror_rng = ops.CoinFlip()
        self.tofloat = ops.Cast(dtype=types.DALIDataType.FLOAT)
        self.mirror = ops.Flip()
        self.tobool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.norm = ops.Normalize(mean=0.45, stddev=0.226)
        
        self.rotate = ops.Rotate(
            keep_size=True, fill_value=0, interp_type=types.DALIInterpType.INTERP_NN
            )
        self.resize = ops.Resize(
            size=size, mode="not_larger", interp_type=types.DALIInterpType.INTERP_NN
            )
        self.shear = ops.WarpAffine(
            fill_value=0, interp_type=types.DALIInterpType.INTERP_NN
            )

    def iter_setup(self):

        shear_matrix = np.zeros((self.batch_size, 2, 3), dtype=np.float32)
        sx = (np.random.rand() - 0.5) * 2 * self.shear_range[0]
        sy = (np.random.rand() - 0.5) * 2 * self.shear_range[1]

        shear_matrix[:, 0, 0] = 1.0 / (1 - sx * sy)
        shear_matrix[:, 1, 1] = 1.0 / (1 - sx * sy)
        shear_matrix[:, 0, 1] = -1.0 * sx / (1 - sx * sy)
        shear_matrix[:, 1, 0] = -1.0 * sy / (1 - sx * sy)

        self.feed_input(self.affine, shear_matrix)

        coin = np.ones((self.batch_size, 1), dtype=np.uint8) * np.random.randint(2)
        self.feed_input(self.coin, coin)

    def define_graph(self):

        jpeg, mask, num, ph = self.input()
        frames = self.decode(jpeg)
        frames = self.tofloat(frames)
        # noise = self.noise_rng()
        # contrast = self.contrast_rng()
        angle = self.rotate_rng()

        # frames = frames + noise
        # frames = frames * contrast
        # mask = mask.gpu()

        self.affine = self.affine_source()
        self.coin = self.mirror_rng()
        frames, mask = self.shear(frames, self.affine), self.shear(mask, self.affine)

        frames, mask = self.rotate(frames, angle=angle), self.rotate(mask, angle=angle)
        frames, mask = self.resize(frames), self.resize(mask)
        hframes, hmask = self.mirror(frames), self.mirror(mask)
        m = self.tobool(self.coin)

        frames = m * frames + (m ^ True) * hframes
        mask = m * mask + (m ^ True) * hmask

        frames = frames / 255.0
        frames = self.norm(frames)

        return (frames, mask, num, ph)

class DaliPytorchLoader(object):

    def __init__(self, datasets, data_freq, batch_size, size, device_id, num_workers):

        self.freqs = data_freq
        self.loaders = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device_id = device_id
        self.size = size

        acc = 0
        datalist = datasets.datasets
        for freq in data_freq:
            self.loaders.append(DaliLoader(datalist[acc], freq))
            acc += freq

    def __iter__(self):

        if hasattr(self, 'pipes'):
            del self.pipes

        if hasattr(self, 'iterators'):
            del self.iterators

        self.pipes = [
            TrainPipeline(
                reader=loader,
                batch_size=self.batch_size,
                device_id=self.device_id,
                size=self.size,
                num_threads=self.num_workers
            )
            for loader in self.loaders
        ]

        for pipe in self.pipes:
            pipe.build()

        # names = [loader.datasets.__name__ for loader in self.loaders]
        self.iterators = [
            DALIGenericIterator(
                pipelines=[pipe], 
                output_map=['frames', 'masks', 'num_objects', 'max_obj'],
                dynamic_shape=True,
                size=len(loader) * self.batch_size
            )
            for pipe, loader in zip(self.pipes, self.loaders)
        ]

        it_idx = list(range(len(self.iterators)))
        random.shuffle(it_idx)
        for iid in it_idx:
            for data in self.iterators[iid]:
                # preprocess the object
                frames = data[0]['frames'].unsqueeze(0)
                masks = data[0]['masks']
                objs = data[0]['num_objects'][0]
                max_obj = int(data[0]['max_obj'][0].item())

                num_obj = int(objs.item())

                if num_obj == 0:
                    continue

                time = masks.shape[0]

                sampled_idx = random.sample(range(1, num_obj+1), min(MAX_TRAINING_OBJ, num_obj))
                sampled_idx.sort()
                
                for idx in range(time):
                    msk = masks[idx]
                    new_anno = msk.new_zeros(msk.shape)
                    for new_id, old_id in enumerate(sampled_idx):
                        new_anno[msk==old_id] = new_id + 1
                        
                    masks[idx] = new_anno

                num_obj = len(sampled_idx)
                
                # if torch.sum(masks[0] == 1) == 0:
                #     print('pass')
                #     vis(frames, masks, num_obj)
                #     continue

                masks = convert_mask(masks, max_obj).unsqueeze(0)

                frames = frames.permute(0, 1, 4, 2, 3)
                masks = masks.permute(0, 1, 4, 2, 3)

                yield (frames, masks, [num_obj], max_obj)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


def get_dali_loader(datasets, data_freq, batch_size, size, device_id, num_workers):

    loader = DaliPytorchLoader(datasets, data_freq, batch_size, size, device_id, num_workers)    

    return loader

if __name__ == '__main__':

    d = DaliLoader([0, 1, 2, 3, 4], 2)
    for _ in range(5):
        print('test')
        for i in d:
            print(i)
