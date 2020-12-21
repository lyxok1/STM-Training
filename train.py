from libs.dataset.data import (
        ROOT, 
        MAX_TRAINING_OBJ, 
        build_dataset, 
        multibatch_collate_fn, 
        convert_one_hot, 
        convert_mask
    )

from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, write_mask, save_checkpoint, adjust_learning_rate
from libs.models.models import STAN
from libs.config import getCfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import libs.utils.logger as logger
import libs.dataset.dali as dali

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import cv2
import copy

from progress.bar import Bar
from collections import OrderedDict

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

MAX_FLT = 1e6

opt, _ = parse_args()

# def parse_args():
#     parser = argparse.ArgumentParser('Training Mask Segmentation')
#     parser.add_argument('--gpu', default='', type=str, help='set gpu id to train the network')
#     return parser.parse_args()

# args = parse_args()
# Use GPU
device = 'cuda:{}'.format(opt.gpu_id)

def main():

    # setup
    start_epoch = 0    
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    opt.output = osp.join(osp.join(opt.checkpoint, opt.output_dir))
    if not osp.exists(opt.output):
        os.mkdir(opt.output)

    logfile = osp.join(opt.checkpoint, opt.mode+'_log.txt')
    logger.setup(filename=logfile, resume=opt.resume != '')
    log = logger.getLogger(__name__)

    # Data
    log.info('Preparing dataset')

    input_dim = tuple(opt.input_size)

    train_transformer = TrainTransform(size=input_dim)
    test_transformer = TestTransform(size=input_dim)

    datalist = []
    for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):

        if opt.data_backend == 'DALI' and not dataset.startswith('DALI'):
            dataset = 'DALI' + dataset

        ds = build_dataset(
            name=dataset,
            train=True, 
            sampled_frames=opt.sampled_frames, 
            transform=train_transformer, 
            max_skip=max_skip, 
            samples_per_video=opt.samples_per_video
        )
        datalist += [copy.deepcopy(ds) for _ in range(freq)]

    trainset = data.ConcatDataset(datalist)

    testset = build_dataset(
        name=opt.valset,
        train=False,
        transform=test_transformer,
        samples_per_video=1
        )

    if opt.data_backend == 'PIL':
        trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                      collate_fn=multibatch_collate_fn)
    elif opt.data_backend == 'DALI':
        trainloader = dali.get_dali_loader(trainset, data_freq=opt.datafreq, batch_size=opt.sampled_frames, size=input_dim, 
                                        device_id=opt.gpu_id, num_workers=opt.workers)
    else:
        raise TypeError('unkown data backend {}'.format(opt.data_backend))

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    log.info("creating model")

    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    net.eval()
    if use_gpu:
        net = net.to(device)

    # set training parameters
    for p in net.parameters():
        p.requires_grad = True

    criterion = None
    celoss = cross_entropy_loss

    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj, ref: celoss(pred, target, obj, ref=ref) + mask_iou_loss(pred, target, obj, ref=ref)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    optimizer = None
    
    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                        momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                        betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    # Resume
    title = 'STAN'
    minloss = float('inf')
    max_time = 0.0

    if opt.resume:
        # Load checkpoint.
        log.info('Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        # opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        net.load_param(checkpoint['state_dict'])
        skips = checkpoint['max_skip']
        
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    # trainloader.dataset.datasets[idx].set_max_skip(skip)
                    trainset.datasets[idx].set_max_skip(skip)
            else:
                # trainloader.dataset.set_max_skip(skip)
                trainset.set_max_skip(skips[0])
        except:
            log.warn('Initializing max skip fail')

    else:
        if opt.initial:
            log.info('Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial, map_location=device)
            if isinstance(weight, OrderedDict):
                net.load_param(weight)
            else:
                net.load_param(weight['state_dict'])

        start_epoch = 0

    # Train and val
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)

    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        log.info('Skip Info:')
        skip_info = dict()
        if isinstance(trainset, data.ConcatDataset):
            for dataset in trainset.datasets:
                skip_info.update(
                        {type(dataset).__name__: dataset.max_skip}
                    )
        else:
            skip_info.update(
                    {type(trainset).__name__: dataset.max_skip}
                )

        skip_print = ''
        for k, v in skip_info.items():
            skip_print += '{}: {} '.format(k, v)
        log.info(skip_print)

        train_loss = train(trainloader,
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=use_gpu,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold,
                           backend=opt.data_backend)

        if (epoch + 1) % opt.epoch_per_test == 0:
            time_cost = test(testloader,
                            model=net,
                            criterion=criterion,
                            epoch=epoch,
                            use_cuda=use_gpu,
                            opt=opt)

            log.info('results are saved at {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))
        else:
            time_cost = 0.0

        # append logger file
        log_format = 'Epoch: {} LR: {} Loss: {} Test Time: {}s'
        log.info(log_format.format(epoch+1, opt.learning_rate, train_loss, time_cost))

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainset, data.ConcatDataset):
                for dataset in trainset.datasets: # trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                # trainloader.dataset.increase_max_skip()
                trainset.increase_max_skip()

        # save model
        skips = [ds.max_skip for ds in trainset.datasets]

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'max_skip': skips,
        }, epoch + 1, checkpoint=opt.checkpoint, filename=opt.mode)

    log.info('minimum loss: {:f}'.format(minloss))

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold, backend):
    # switch to train mode

    data_time = AverageMeter()
    loss = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader):

        frames, masks, objs, _ = data
        max_obj = masks.shape[2]-1
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            frames = frames.to(device)
            masks = masks.to(device)

        N, T, C, H, W = frames.size()
        total_loss = 0.0
        for idx in range(N):
            frame = frames[idx]
            mask = masks[idx]
            num_objects = objs[idx]

            keys = []
            vals = []
            scales = []
            for t in range(1, T):
                # memorize
                if t-1 == 0:
                    tmp_mask = mask[t-1:t]
                else:
                    tmp_mask = out                    

                key, val, r4 = model(frame=frame[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

                keys.append(key)
                vals.append(val)

                # segment
                tmp_key = torch.cat(keys, dim=1)
                tmp_val = torch.cat(vals, dim=1)

                logits, ps = model(frame=frame[t:t+1, :, :, :], keys=tmp_key, values=tmp_val,
                  num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)
                gt = mask[t:t+1]
                
                total_loss = total_loss + criterion(out, gt, num_objects, ref=mask[0:1, :num_objects+1])

            # cycle-consistancy
            key, val, r4 = model(frame=frame[T-1:T, :, :, :], mask=out, num_objects=num_objects)
            keys.append(key)
            vals.append(val)

            cycle_loss = 0.0
            for t in range(T-1, 0, -1): 
                cm = np.transpose(mask[t].detach().cpu().numpy(), [1, 2, 0])
                if convert_one_hot(cm, max_obj).max() == num_objects:
                    # tmp_key = torch.cat(keys[1:t+1], dim=1)
                    # tmp_val = torch.cat(vals[1:t+1], dim=1)
                    tmp_key = keys[t]
                    tmp_val = vals[t]
                    # print(tmp_key.shape, tmp_val.shape)
                    logits, ps = model(frame=frame[0:1, :, :, :], keys=tmp_key, values=tmp_val,
                        num_objects=num_objects, max_obj=max_obj)
                    first_out = torch.softmax(logits, dim=1)
                    cycle_loss += criterion(first_out, mask[0:1], num_objects, ref=mask[t:t+1, :num_objects+1])
            
            total_loss = total_loss + cycle_loss

        total_loss = total_loss / (N * (T-1))

        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

            # compute gradient and do SGD step (divided by accumulated steps)
            total_loss /= iter_size
            total_loss.backward()

        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=loss.val
        )
        bar.next()
    bar.finish()

    return loss.avg

def test(testloader, model, criterion, epoch, use_cuda, opt):

    data_time = AverageMeter()

    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
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
            # compute output
            t1 = time.time()

            T, _, H, W = frames.shape
            pred = [masks[0:1]]
            keys = []
            vals = []
            scales = []
            for t in range(1, T):
                if t-1 == 0:
                    tmp_mask = masks[0:1]
                elif 'frame' in info and t-1 in info['frame']:
                    # start frame
                    mask_id = info['frame'].index(t-1)
                    tmp_mask = masks[mask_id:mask_id+1]
                    num_objects = max(num_objects, tmp_mask.max())
                else:
                    tmp_mask = out

                # memorize
                key, val, r4 = model(frame=frames[t-1:t], mask=tmp_mask, num_objects=num_objects)

                # segment TODO: deal with sudden num_object change
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
                # tmp_scale = torch.stack(scales+[scale], dim=1)
                logits, ps = model(frame=frames[t:t+1], keys=tmp_key, values=tmp_val, 
                    num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)
                pred.append(out)

                if (t-1) % opt.save_freq == 0:
                    keys.append(key)
                    vals.append(val)
                    # scales.append(scale)
            
            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            write_mask(pred, info, opt, directory=opt.output_dir)

            toc = time.time() - t1

            data_time.update(toc, 1)
           
            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return data_time.sum

if __name__ == '__main__':
    main()
