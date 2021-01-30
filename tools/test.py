'''
Instructions to use:
     python3.7 tools/test.py --cfg experiments/refuge.yaml
     TRAIN.MV_IDEA True
     TRAIN.MV_IDEA_HM1 True
     TEST.MODEL_FILE output/refuge/fovea_net/refuge/checkpoint_HM1_L7_Aug28.pth
     TEST.RELEASE_TEST False

     python3.7 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best.pth
     cfg.TEST.RELEASE_TEST True

description:
    test the model

'''
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

"""
        Zhang Yu original -- mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
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
        'mean': { 'train':  [0.400, 0.298, 0.229],
                  'test':   [0.590, 0.490, 0.442],
                  'valid':  [0.596, 0.493, 0.440],
                  'valid2': [0.566, 0.416, 0.306] },
        'std':  { 'train':  [0.176, 0.140, 0.108],
                  'test':   [0.184, 0.174, 0.153],
                  'valid':  [0.184, 0.174, 0.152],
                  'valid2': [0.184, 0.159, 0.140] },

    before CLAHE
    train:
        mean = [0.084, 0.168, 0.282], std = [0.062, 0.110, 0.189]
        mean = [0.282, 0.168, 0.084], std = [0.189, 0.110, 0.062]
    val:
        mean = [0.215, 0.270, 0.409], std = [0.160, 0.203, 0.288]
        mean = [0.409, 0.270, 0.215], std = [0.288, 0.203, 0.160]
    test:
        mean = [0.222, 0.271, 0.404], std = [0.163, 0.202, 0.284]
        mean = [0.404, 0.271, 0.222], std = [0.284, 0.202, 0.163]
    val2:
        mean = [0.059, 0.208, 0.417], std = [0.079, 0.152, 0.273]
        mean = [0.417, 0.208, 0.059], std = [0.273, 0.152, 0.079]

    after CLAHE
        train: 
        mean = [0.134, 0.227, 0.316], std = [0.089, 0.142, 0.197]

        val:
        mean = [0.257, 0.303, 0.390], std = [0.186, 0.219, 0.268]

        test:
        mean = [0.262, 0.303, 0.386], std = [0.188, 0.218, 0.264]

        val2:
        mean = [0.128, 0.283, 0.422], std = [0.135, 0.187, 0.262]

"""

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
        (filepath, tempfilename) = os.path.split(cfg.TEST.MODEL_FILE)
        if "checkpoint" in tempfilename:
            # workaround to load python2 model -- Note: the final result could be different
            if 'P2' in tempfilename:
                checkpoint = torch.load(cfg.TEST.MODEL_FILE, encoding='latin1')
            else:
                checkpoint = torch.load(cfg.TEST.MODEL_FILE)
            model.load_state_dict(checkpoint['best_state_dict'])
        else:
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
    release_result = cfg.TEST.RELEASE_TEST
    if release_result is False:
        normalize = transforms.Normalize(
            # mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
            # mean = [0.404, 0.271, 0.222], std = [0.284, 0.202, 0.163]
            # 160 images from refuge1 val
            mean=[0.404, 0.267, 0.213], std=[0.285, 0.201, 0.159]
        )
        valid_dataset = importlib.import_module('dataset.'+cfg.DATASET.DATASET).Dataset(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        trial_enable = cfg.TEST.TRIAL_RUN
        if not trial_enable:
            normalize = transforms.Normalize(
                # mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
                mean = [0.417, 0.208, 0.059], std = [0.273, 0.152, 0.079]
            )
        else:
            normalize = transforms.Normalize(
                # prepare for refuge2 final submission - 'Refuge2-Ext'
                # mean=[0.435, 0.211, 0.070], std=[0.310, 0.166, 0.085]
                mean=[0.532, 0.259, 0.086], std=[0.259, 0.148, 0.088]
            )

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

    debug_enable = cfg.TEST.DEBUG
    # evaluate on validation set
    if debug_enable:
        debug_dir = './debug'
        if not os.path.isdir(str(debug_dir)):
            os.mkdir(debug_dir)
        validate(cfg, valid_loader, valid_dataset, model, criterion, debug_dir, tb_log_dir, debug_all=debug_enable)
    else:
        validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir)



if __name__ == '__main__':
    main()
    print("Refuge Fovea Test Program Exit ... \n")
