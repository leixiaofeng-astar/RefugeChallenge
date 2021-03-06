from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import argparse
import os
import pprint
import shutil
import importlib
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime

from config.common import _C as cfg
from config.common import update_config
from core.loss import HybridLoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models


def timer(start_time=None):
    if not start_time:
        return datetime.now()
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.\n' % (thour, tmin, round(tsec, 2)))
        return datetime.now()

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
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
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_builder = importlib.import_module("models." + cfg.MODEL.NAME).get_fovea_net
    model = model_builder(cfg, is_train=True)

    # xiaofeng add for load parameter
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)

    # copy model file -- xiaofeng comment it
    # this_dir = os.path.dirname(__file__)
    # shutil.copy2(os.path.join(this_dir, '../models', cfg.MODEL.NAME + '.py'), final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = HybridLoss(
        roi_weight=cfg.LOSS.ROI_WEIGHT,
        regress_weight=cfg.LOSS.REGRESS_WEIGHT,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
        hrnet_only=cfg.TRAIN.HRNET_ONLY).cuda()

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
    # )
    # train_dataset = importlib.import_module('dataset.'+cfg.DATASET.DATASET).Dataset(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_dataset = importlib.import_module('dataset.'+cfg.DATASET.DATASET).Dataset(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # 
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=cfg.TRAIN.SHUFFLE,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )

    db_trains = []
    db_vals = []
    final_full_test = cfg.TRAIN.FULL_DATA
    normalize_1 = transforms.Normalize(
        mean=[0.282, 0.168, 0.084], std=[0.189, 0.110, 0.062]
    )
    train_dataset_1 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET_1, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize_1,
        ])
    )
    db_trains.append(train_dataset_1)

    normalize_2 = transforms.Normalize(
        mean = [0.409, 0.270, 0.215], std = [0.288, 0.203, 0.160]
    )
    train_dataset_2 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET_2, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize_2,
        ])
    )
    db_trains.append(train_dataset_2)

    if final_full_test is True:
        normalize_3 = transforms.Normalize(
            mean = [0.404, 0.271, 0.222], std = [0.284, 0.202, 0.163]
        )
        train_dataset_3 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize_3,
            ])
        )
        db_trains.append(train_dataset_3)

    train_dataset = ConcatDataset(db_trains)
    logger.info("Combined Dataset: Total {} images".format(len(train_dataset)))

    train_batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    normalize = transforms.Normalize(
        mean=[0.404, 0.271, 0.222], std=[0.284, 0.202, 0.163]
    )
    val_dataset_1 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    db_vals.append(val_dataset_1)

    if final_full_test is True:
        normalize_1 = transforms.Normalize(
            mean=[0.282, 0.168, 0.084], std=[0.189, 0.110, 0.062]
        )
        val_dataset_2 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET_1, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize_1,
            ])
        )
        db_vals.append(val_dataset_2)

        normalize_2 = transforms.Normalize(
            mean=[0.409, 0.270, 0.215], std=[0.288, 0.203, 0.160]
        )
        val_dataset_3 = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET_2, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize_2,
            ])
        )
        db_vals.append(val_dataset_3)

    valid_dataset = ConcatDataset(db_vals)

    logger.info("Val Dataset: Total {} images".format(len(valid_dataset)))

    test_batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    logger.info("Train len: {}, batch_size: {}; Test len: {}, batch_size: {}" \
                .format(len(train_loader), train_batch_size, len(valid_loader), test_batch_size))

    best_metric = 1e6
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if cfg.TEST.MODEL_FILE:
        checkpoint_file = cfg.TEST.MODEL_FILE
    else:
        checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        # begin_epoch = checkpoint['epoch']
        begin_epoch = 0   # xiaofeng change it
        best_metric = checkpoint['metric']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if cfg.TRAIN.LR_EXP:
        # llr=lr∗gamma∗∗epoch
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.TRAIN.GAMMA1, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        start_time = timer()

        lr_scheduler.step()

        # evaluate on validation set
        # lr_metric, hr_metric, final_metric = validate(
        #     cfg, valid_loader, valid_dataset, model, criterion,
        #     final_output_dir, tb_log_dir, writer_dict, db_vals
        # )
        # print("validation before training spent time:")
        # timer(start_time)  # timing ends here for "start_time" variable

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        print("epoch %d train spent time:" % (epoch))
        train_time = timer(start_time)  # timing ends here for "start_time" variable

        # if epoch >= int(cfg.TRAIN.END_EPOCH/10):
        # evaluate on validation set
        lr_metric, hr_metric, final_metric = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict, db_vals
        )

        print("validation spent time:")
        val_time = timer(train_time)  # timing ends here for "start_time" variable

        min_metric = min(lr_metric, hr_metric, final_metric)
        if min_metric <= best_metric:
            best_metric = min_metric
            best_model = True
            logger.info('=> epoch [{}] best model result: {}'.format(epoch, best_metric))
        else:
            best_model = False

        # xiaofeng changed it
        if best_model is True:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            # transfer the model to CPU before saving to fix unstable bug:
            # github.com/pytorch/pytorch/issues/10577

            model = model.cpu()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'metric': final_metric,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
            model = model.cuda()

            print("saving spent time:")
            end_time = timer(val_time)  # timing ends here for "start_time" variable
        elif (epoch % 60 == 0) and (epoch != 0):
            logger.info('=> saving epoch {} checkpoint to {}'.format(epoch, final_output_dir))
            # transfer the model to CPU before saving to fix unstable bug:
            # github.com/pytorch/pytorch/issues/10577

            time_str = time.strftime('%Y-%m-%d-%H-%M')
            if cfg.TRAIN.HRNET_ONLY:
                checkpoint_filename = 'checkpoint_HRNET_epoch%d_%s.pth' % (epoch, time_str)
            else:
                checkpoint_filename = 'checkpoint_Hybrid_epoch%d_%s.pth' % (epoch, time_str)
            model = model.cpu()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'metric': final_metric,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, checkpoint_filename)
            model = model.cuda()

    # xiaofeng change
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if cfg.TRAIN.HRNET_ONLY:
        model_name = 'final_state_HRNET_%s.pth' % (time_str)
    else:
        model_name = 'final_state_Hybrid_%s.pth' % (time_str)

    final_model_state_file = os.path.join(final_output_dir, model_name)
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    # save a final checkpoint
    model = model.cpu()
    save_checkpoint({
        'epoch': epoch + 1,
        'model': cfg.MODEL.NAME,
        'state_dict': model.state_dict(),
        'best_state_dict': model.module.state_dict(),
        'metric': final_metric,
        'optimizer': optimizer.state_dict(),
    }, best_model, final_output_dir, "checkpoint_final_state.pth")
    # model = model.cuda()


if __name__ == '__main__':
    main()
    print("Refuge Fovea Train Program Exit ... \n")