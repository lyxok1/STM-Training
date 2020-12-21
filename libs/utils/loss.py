import math
import torch

from .utility import mask_iou

def binary_entropy_loss(pred, target, num_object, eps=0.001, ref=None):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

    loss = torch.mean(ce)

    # TODO: training with bootstrapping

    return loss

def cross_entropy_loss(pred, mask, num_object, bootstrap=0.4, ref=None):

    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape

    pred = -1 * torch.log(pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)
    ce = pred[:, :num_object+1] * mask[:, :num_object+1]
    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def mask_iou_loss(pred, mask, num_object, ref=None):

    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1

    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0

    for i in range(N):
        obj_loss = (1.0 - mask_iou(pred[i, start:num_object+start], mask[i, start:num_object+start], averaged=False))
        if ref is not None:
            obj_loss = obj_loss[valid[i, start:]]

        loss += torch.mean(obj_loss)

    loss = loss / N
    return loss

def smooth_l1_loss(pred, target, gamma=0.075):

    diff = torch.abs(pred-target)
    diff[diff>gamma] -= gamma / 2
    diff[diff<=gamma] *= diff[diff<=gamma] / (2 * gamma)

    return torch.mean(diff)





