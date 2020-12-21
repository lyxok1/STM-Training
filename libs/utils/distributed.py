import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import builtins

import sys
import os

from .logger import getLogger

logger = getLogger(name=__name__)

def _silent_print(*args, **kwargs):
    pass

def setup(local_rank, opt):

    logger.info('setup distributed process')
    num_proc = int(os.getenv('WORLD_SIZE', default=len(opt.multi_gpu_ids)))

    assert num_proc <= len(opt.multi_gpu_ids)

    dist.init_process_group(
        backend=opt.backend,
        init_method=opt.init_method,
        world_size=num_proc,
        rank=local_rank
        )

    if not torch.cuda.is_available():
        raise NotImplementedError('gpu is not enabled on current device')
    else:
        torch.cuda.set_device(opt.multi_gpu_ids[local_rank])

    if not is_master_proc():
        # suppress stdout
        builtins.print = _silent_print

def is_master_proc():

    if not (dist.is_available() and dist.is_initialized()):
        raise NotImplementedError('distributed training not supported')
    else:
        return dist.get_rank() == 0

def get_local_rank():

    if not (dist.is_available() and dist.is_initialized()):
        raise NotImplementedError('distributed training not supported')
    else:
        return dist.get_rank()

def get_world_size():

    if not (dist.is_available() and dist.is_initialized()):
        raise NotImplementedError('distributed training not supported')
    else:
        return dist.get_world_size()

def sync():

    dist.barrier()

def sync_tensor(tensor):

    if not (dist.is_available() and dist.is_initialized()):
        raise NotImplementedError('distributed training not supported')
    else:
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()

    return tensor

def dispatch_job(opt, target):

    num_proc = len(opt.multi_gpu_ids)
    mp.spawn(
        fn=target,
        nprocs=num_proc,
        args=(opt,)
        )