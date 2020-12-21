from yacs.config import CfgNode

OPTION = CfgNode()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['DAVIS17', 'VOS']
OPTION.valset = 'DAVIS17'
OPTION.datafreq = [5, 1]
OPTION.input_size = (240, 427)   # input image size for training
OPTION.sampled_frames = 3        # min sampled time length while trianing
OPTION.max_skip = [5, 3]       # max skip time length while trianing
OPTION.samples_per_video = 1    # sample numbers per video
OPTION.data_backend = 'PIL'     # dataloader backend 'PIL' or 'DALI'

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.arch = 'resnet50'
OPTION.save_freq = 5
OPTION.epochs_per_increment = 5
OPTION.split_k = 3

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 240
OPTION.train_batch = 1
OPTION.learning_rate = 0.00001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adam'             # 'sgd' or 'adam'
OPTION.weight_decay = 5e-4
OPTION.iter_size = 4
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training
OPTION.alpha = 1.25
OPTION.lambdas = [1.0, 0.4]

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 1
OPTION.correction_rate = 150
OPTION.correction_momentum = 0.9
OPTION.loop = 10

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'ckpt'
OPTION.initial = ''      # path to initialize the backbone
OPTION.resume = ''       # path to restart from the checkpoint
OPTION.video_path = ''   # path to video on which the model is running
OPTION.mask_path = ''    # path to mask on withc the model is running
OPTION.gpu_id = 1
OPTION.workers = 1
OPTION.save_indexed_format = 'index'
OPTION.output_dir = 'output'

# ------------------------------------------- optional distributed configuration-------------------------------
OPTION.multi_gpu_ids = [0, 1, 2, 3]
OPTION.backend = 'nccl'
OPTION.init_method = 'env://'

def sanity_check(opt):

    assert isinstance(opt.trainset, (str, list)), \
        'training set should be specified by a string or string list'
    assert isinstance(opt.valset, str), \
        'validation set should be a single dataset'
    assert opt.data_backend in ['PIL', 'DALI'], \
        'only PIL or DALI backend are supported'
    assert opt.solver in ['adam', 'sgd']
    assert opt.mode in ['mask', 'threshold', 'recurrent']
    assert opt.loss in ['iou', 'ce', 'both']
    assert opt.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101']

def getCfg():

    return OPTION.clone()