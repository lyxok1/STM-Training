import numpy as np
import math

import torch
import os
import os.path as osp
import shutil
import cv2
import random
import argparse

from PIL import Image
from ..dataset.data import ROOT
from ..config import getCfg, sanity_check
from .logger import getLogger

logger = getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--cfg', default='', type=str, help='path to config file')
    parser.add_argument('--local_rank', default=-1, type=int, help='process local rank, only used for distributed training')
    parser.add_argument('options', help='other configurable options', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    opt = getCfg()

    if osp.exists(args.cfg):
        opt.merge_from_file(args.cfg)

    if len(args.options) > 0:
        assert len(args.options) % 2 == 0, 'configurable options must be key-val pairs'
        opt.merge_from_list(args.options)

    sanity_check(opt)

    return opt, args.local_rank

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint'):
    
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    logger.info('save model at {}'.format(filepath))

def write_mask(mask, info, opt, directory='results'):

    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    directory = os.path.join(ROOT, directory)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])    

        if opt.save_indexed_format == 'index':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')

        elif opt.save_indexed_format == 'segmentation':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max()+1):
                seg[rescale_mask==k, :] = info['palette'][(k*3):(k+1)*3][::-1]

            inp_img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'heatmap':
            
            rescale_mask[rescale_mask<0] = 0.0
            rescale_mask = np.max(rescale_mask[:, :, 1:], axis=2)
            rescale_mask = (rescale_mask - rescale_mask.min()) / (rescale_mask.max() - rescale_mask.min()) * 255
            seg = rescale_mask.astype(np.uint8)
            # seg = cv2.GaussianBlur(seg, ksize=(5, 5), sigmaX=2.5)

            seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            inp_img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'mask':

            fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)

            seg = np.zeros((h, w, 3), dtype=np.uint8)
            seg[fg==1, :] = info['palette'][3:6][::-1]

            inp_img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        else:
            raise TypeError('unknown save format {}'.format(opt.save_indexed_format))
        

def mask_iou(pred, target, averaged = True):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = inter / union

    if averaged:
        iou = torch.mean(iou)

    return iou

def adjust_learning_rate(optimizer, epoch, opt):

    if epoch in opt.milestone:
        opt.learning_rate *= opt.gamma
        for pm in optimizer.param_groups:
            pm['lr'] *= opt.learning_rate

def pointwise_dist(points1, points2):

    # compute the point-to-point distance matrix

    N, d = points1.shape
    M, _ = points2.shape

    p1_norm = torch.sum(points1**2, dim=1, keepdim=True).expand(N, M)
    p2_norm = torch.sum(points2**2, dim=1).unsqueeze(0).expand(N, M)
    cross = torch.matmul(points1, points2.permute(1, 0))

    dist = p1_norm - 2 * cross + p2_norm

    return dist

def furthest_point_sampling(points, npoints):

    """
    points: [N x d] torch.Tensor
    npoints: int

    """
    
    old = 0
    output_idx = []
    output = []
    dist = pointwise_dist(points, points)
    fdist, fidx = torch.sort(dist, dim=1, descending=True)

    for i in range(npoints):
        fp = 0
        while fp < points.shape[0] and fidx[old, fp] in output_idx:
            fp += 1

        old = fidx[old, fp]
        output_idx.append(old)
        output.append(points[old])

    return torch.stack(output, dim=0)

def split_mask_by_k(mask, k):

    """
    mask: [H x W] torch.Tensor (one-hot encoded or float)
    k: int

    ret: [k x H x W] torch.Tensor
    """

    if k == 0:
        return mask.unsqueeze(0)

    H, W = mask.shape
    meshx = torch.Tensor([[i for i in range(W)]]).float().to(mask.device).expand(H, W)
    meshy = torch.Tensor([[i] for i in range(H)]).float().to(mask.device).expand(H, W)
    mesh = torch.stack([meshx, meshy], dim=2)

    foreground = mesh[mask>0.5, :].view(-1, 2)

    # samples = furthest_point_sampling(foreground, k)

    npoints = foreground.shape[0]
    nidx = random.sample(range(npoints), k)
    samples = foreground[nidx, :]

    mesh = mesh.view(-1, 2)
    dist = pointwise_dist(mesh, samples)
    _, cidx = torch.min(dist, dim=1)
    cidx = cidx.view(H, W)

    output = []

    for i in range(k):
        output.append(((cidx == i) * (mask > 0.5)).float())

    return torch.stack(output, dim=0)

def mask_to_box(masks, num_objects):

    """
    convert a mask annotation to coarse box annotation

    masks: [N x (K+1) x H x W]
    """

    N, K, H, W = masks.shape
    output = masks.new_zeros(masks.shape)

    for n in range(N):
        for o in range(1+num_objects):
            for start_x in range(W):
                if torch.sum(masks[n, o, :, start_x]) > 0:
                    break

            for end_x in range(W-1, -1, -1):
                if torch.sum(masks[n, o, :, end_x]) > 0:
                    break

            for start_y in range(H):
                if torch.sum(masks[n, o, start_y, :]) > 0:
                    break

            for end_y in range(H-1, -1, -1):
                if torch.sum(masks[n, o, end_y, :]) > 0:
                    break

            if start_x <= end_x and start_y <= end_y:
                output[n, o, start_y:end_y+1, start_x:end_x+1] = 1

    return output