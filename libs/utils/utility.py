import numpy as np
import math

import torch
import os
import shutil
import cv2
import random

from PIL import Image
from ..dataset.data import ROOT

def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    print('==> save model at {}'.format(filepath))
    if is_best:
        cpy_file = os.path.join(checkpoint, filename+'_model_best.pth.tar')
        shutil.copyfile(filepath, cpy_file)
        print('==> save best model at {}'.format(cpy_file))

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

    if 'frame' not in info:
        min_t = 0
        step = 1
    else:
        min_t = min(info['frame'])
        step = 5

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        output_name = '{:0>5d}.png'.format(t * step + min_t)
        if opt.save_indexed_format:
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')
        else:
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max()+1):
                seg[rescale_mask==k, :] = info['palette'][(k*3):(k+1)*3]
            inp_img = cv2.imread(os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)
        

def mask_iou(pred, target):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum(inter / union) / N

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