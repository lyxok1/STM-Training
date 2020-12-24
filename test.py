from libs.dataset.data import ROOT, build_dataset, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, write_mask, save_checkpoint, adjust_learning_rate, mask_iou
from libs.models.models import STAN
from libs.config import getCfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import libs.utils.logger as logger

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import cv2
import argparse

from progress.bar import Bar
from collections import OrderedDict

# from options import OPTION as opt

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

MAX_FLT = 1e6

# parse args

opt, _ = parse_args()

# Use CUDA
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

logger.setup(filename='test_out.log', resume=False)
log = logger.getLogger(__name__)

def main():

    # Data
    log.info('Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)
    testset = build_dataset(
        name=opt.valset,
        train=False, 
        transform=test_transformer, 
        samples_per_video=1
        )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    log.info("Creating model")

    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # set eval to freeze batchnorm update
    net.eval()

    if use_gpu:
        net.to(device)

    # set training parameters
    for p in net.parameters():
        p.requires_grad = False

    # Resume
    title = 'STAN'

    if opt.initial:
        # Load checkpoint.
        log.info('Loading weights from checkpoint {}'.format(opt.initial))
        assert os.path.isfile(opt.initial), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial, map_location=device)
        try:
            net.load_param(checkpoint['state_dict'])
        except:
            net.load_param(checkpoint)

    # Train and val
    log.info('Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))

    test(testloader,
        model=net,
        use_cuda=use_gpu,
        opt=opt)

    log.info('Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))

def test(testloader, model, use_cuda, opt):

    data_time = AverageMeter()
    criterion = lambda pred, target, obj: cross_entropy_loss(pred, target, obj) + mask_iou_loss(pred, target, obj)

    # with torch.no_grad():
    for p in model.parameters():
        p.requires_grad = False

    for batch_idx, data in enumerate(testloader):

        frames, masks, objs, infos = data

        if use_cuda:
            frames = frames.to(device)
            masks = masks.to(device)
            
        frames = frames[0]
        masks = masks[0]
        num_objects = objs[0]
        info = infos[0]
        max_obj = masks.shape[1]-1
        T, _, H, W = frames.shape

        prev = frames.new_zeros((1, 3, H, W))
        prev_mask = frames.new_zeros((1, num_objects+1, H, W))

        bar = Bar('video {}: {}'.format(batch_idx+1, info['name']), max=T-1)
        log.info('Runing video {}, objects {:d}'.format(info['name'], num_objects))
        # compute output
        
        pred = [masks[0:1]]
        keys = []
        vals = []
        ref_mask = None
        for t in range(1, T):
            with torch.no_grad():
                if t-1 == 0:
                    tmp_mask = masks[0:1]
                    ref_mask = tmp_mask
                    ref_index = 0
                elif 'frame' in info and t-1 < len(info['frame']['imgs']) and info['frame']['imgs'][t-1] in info['frame']['masks']:
                    # start frame
                    mask_stamp = info['frame']['imgs'][t-1]
                    mask_id = info['frame']['masks'].index(mask_stamp)
                    tmp_mask = masks[mask_id:mask_id+1]
                    num_objects = max(num_objects, tmp_mask.max())
                    ref_mask = tmp_mask
                    ref_index = t-1
                    # pred[-1] = tmp_mask
                else:
                    tmp_mask = out
                    # tmp_mask = masks[t-1:t]

                t1 = time.time()
                # memorize
                key, val, _ = model(frame=frames[t-1:t], mask=tmp_mask, num_objects=num_objects)

                # segment
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
                logits, ps = model(frame=frames[t:t+1], keys=tmp_key, values=tmp_val, 
                    num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)

                if (t-1) % opt.save_freq == 0:

                    keys.append(key)
                    vals.append(val)

            pred.append(out)
            if t % opt.save_freq == 0:
                # internal loop
                for m in range(opt.loop):
                    tmp_mask = out.detach().clone()
                    tmp_mask.requires_grad = True
                    key, val, _ = model(frame=frames[t:t+1], mask=tmp_mask, num_objects=num_objects)
                    flogits, _ = model(frame=frames[ref_index:ref_index+1], keys=key, values=val,
                     num_objects=num_objects, max_obj=max_obj)

                    fout = torch.softmax(flogits, 1)

                    loss = criterion(fout, ref_mask, num_objects)
                    
                    loss.backward()
                    out = out - opt.correction_rate * tmp_mask.grad

            toc = time.time() - t1
            data_time.update(toc, 1)

            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=t,
                size=T-1,
                data=data_time.sum
            )
            bar.next()
        bar.finish()
            
        pred = torch.cat(pred, dim=0)
        pred = pred.detach().cpu().numpy()
        write_mask(pred, info, opt, directory=opt.output_dir)

    return

if __name__ == '__main__':
    main()
