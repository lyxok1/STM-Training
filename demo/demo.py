from libs.dataset.data import ROOT, convert_mask
from libs.dataset.transform import TestTransform
from libs.utils.loss import *
from libs.utils.utility import parse_args, write_mask, save_checkpoint, adjust_learning_rate, mask_iou
from libs.models.models import STAN
from libs.config import getCfg

import libs.utils.logger as logger

import torch
import torch.nn as nn

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import cv2

from PIL import Image
from progress.bar import Bar

MAX_FLT = 1e6
opt, _ = parse_args()

# Use CUDA
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

logger.setup(filename='demo/demo.log', resume=False)
log = logger.getLogger(__name__)

def main():

    # Data
    log.info('Loading videos from %s' % opt.video_path)

    frame_list = os.listdir(opt.video_path)
    frame_list.sort(key=lambda x: int(x.split('.')[0]))

    video_frames = [np.array(Image.open(osp.join(opt.video_path, name)))  
        for name in frame_list if '.jpg' in name]

    if len(video_frames) <= 1:
        log.error('empty video')
        exit()

    nframes = len(video_frames)
    log.info('Loading total %d frames' % nframes)
    log.info('Loading mask from %s' % opt.mask_path)

    initial_mask = Image.open(osp.join(opt.mask_path))
    palette = initial_mask.getpalette()
    initial_mask = np.array(initial_mask)

    num_obj = initial_mask.max()
    # append with placeholder
    pred = [initial_mask]
    initial_mask = convert_mask(np.array(Image.open(osp.join(opt.mask_path))), num_obj)
    initial_masks = [initial_mask] * nframes

    assert video_frames[0].shape[:2] == initial_mask.shape[:2], 'the mask size mismatches the video frames'
    
    frame_h, frame_w = video_frames[0].shape[:2]
    input_h, input_w = opt.input_size

    sf = min(input_h / frame_h, input_w / frame_w)
    scale_h, scale_w = int(frame_h * sf), int(frame_w * sf)
    p_top, p_left = (input_h - scale_h) // 2, (input_w - scale_w) // 2

    test_transformer = TestTransform(size=tuple(opt.input_size))
    input_frames = [img.copy() for img in video_frames]
    frames, masks = test_transformer(input_frames, initial_masks)

    # Model
    log.info("Creating model")

    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # set eval to freeze batchnorm update
    criterion = lambda pred, target, obj: cross_entropy_loss(pred, target, obj) + mask_iou_loss(pred, target, obj)
    net.eval()

    if use_gpu:
        net.to(device)
        frames = frames.to(device)
        masks = masks.to(device)

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
    log.info('Runing model on input video and mask')

    bar = Bar('processing', max=nframes-1)
    data_time = logger.AverageMeter()
            
    keys = []
    vals = []
    for t in range(1, nframes):
        with torch.no_grad():
            tmp_mask = masks[t-1:t] if t-1 == 0 else out

            t1 = time.time()
            # memorize
            key, val, _ = net(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_obj)

            # segment
            tmp_key = torch.cat(keys+[key], dim=1)
            tmp_val = torch.cat(vals+[val], dim=1)
            logits, ps = net(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_obj, max_obj=num_obj)

            # out = torch.cat([1-ps, ps], dim=0)
            # out = torch.softmax(logits[0], dim=0)
            out = torch.softmax(logits, dim=1)

            if (t-1) % opt.save_freq == 0:
                keys.append(key)
                vals.append(val)

        masks[t] = out[0]
        if t % opt.save_freq == 0:
            # internal loop
            
            for m in range(opt.loop):
                tmp_mask = out.detach().clone()
                tmp_mask.requires_grad = True
                key, val, _ = net(frame=frames[t:t+1], mask=tmp_mask, num_objects=num_obj)
                flogits, _ = net(frame=frames[0:1], keys=key, values=val,
                 num_objects=num_obj, max_obj=9)

                fout = torch.softmax(flogits, 1)

                loss = criterion(fout, masks[0:1], num_obj)
                loss.backward()

                out = out - opt.correction_rate * tmp_mask.grad

        cout = out[0].cpu().detach().numpy()
        m = cout[:, p_top:p_top + input_h, p_left:p_left + input_w].transpose([1, 2, 0])
        scale_mask = cv2.resize(m, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        scale_mask = np.argmax(scale_mask, axis=2)
        pred.append(scale_mask)

        toc = time.time() - t1
        data_time.update(toc, 1)

        # plot progress
        bar.suffix  = '({frame}/{size}) Time: {data:.3f}s'.format(
            frame=t,
            size=nframes-1,
            data=data_time.sum
        )
        bar.next()
    bar.finish()
    
    output_dir = opt.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    video_name = osp.basename(opt.video_path)
    output_dir = osp.join(output_dir, video_name)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    for idx, msk in enumerate(pred):
        img = video_frames[idx][:, :, ::-1]
        canvas = np.zeros(img.shape, dtype=np.uint8)
        for oid in range(1, num_obj + 1):
            canvas[msk==oid, :] = palette[3*oid:(oid+1)*3][::-1]
            masked_img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0.0)
            cv2.imwrite(osp.join(output_dir, '{:0>5}.jpg'.format(idx)), masked_img)

    log.info('Results are saved at: {}'.format(output_dir))

if __name__ == '__main__':
    main()
