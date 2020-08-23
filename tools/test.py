
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import importlib

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config.common import _C as cfg
from config.common import update_config
from core.loss import HybridLoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_builder = importlib.import_module("models." + cfg.MODEL.NAME).get_fovea_net
    model = model_builder(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = HybridLoss(
        roi_weight=cfg.LOSS.ROI_WEIGHT,
        regress_weight=cfg.LOSS.REGRESS_WEIGHT,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    """
        before gray conv
        'mean': { 'train':  [0.449, 0.260, 0.130],
          'test':   [0.654, 0.458, 0.364],
          'valid':  [0.662, 0.459, 0.356],
          'valid2': [0.684, 0.384, 0.163] },
        'std':  { 'train':  [0.240, 0.151, 0.086],
          'test':   [0.215, 0.185, 0.144],
          'valid':  [0.217, 0.185, 0.144],
          'valid2': [0.210, 0.157, 0.127] },
          
        after gray conv
        Calculating /home/user/eye/xf_refuge/data/REFUGE-Training400/Training400/...
        mean:
        [0.138, 0.180, 0.237]
        std:
        [0.090, 0.117, 0.155]
        Calculating /home/user/eye/xf_refuge/data/REFUGE-Validation400/...
        mean:
        [0.260, 0.288, 0.357]
        std:
        [0.189, 0.212, 0.254]
        Calculating /home/user/eye/xf_refuge/data/REFUGE-Test400/...
        mean:
        [0.263, 0.288, 0.354]
        std:
        [0.190, 0.211, 0.251]
        Calculating /home/user/eye/xf_refuge/data/Refuge2-Validation/...
        mean:
        [0.156, 0.231, 0.335]
        std:
        [0.121, 0.164, 0.224]

    """
    normalize = transforms.Normalize(
        # xiaofeng change it for test set
        # mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]  # original one

        # test set
        # mean = [0.263, 0.288, 0.354], std = [0.190, 0.211, 0.251]
        # val2 set
        mean = [0.156, 0.231, 0.335], std = [0.121, 0.164, 0.224]

    )
    # dataset.refuge.Dataset class -- (self, cfg, root, image_set, is_train, transform=None)
    # xiaofeng test
    # valid_dataset = importlib.import_module('dataset.'+cfg.DATASET.DATASET).Dataset(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
    print("Refuge Fovea Test Program Exit ... \n")
