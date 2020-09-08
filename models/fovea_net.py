
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy
import numpy as np

from core.inference import get_max_preds, get_heatmap_center_preds
from utils.transforms import crop_and_resize
from efficientnet.model import EfficientNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

bb2feat_dims = { 'resnet34':  [64, 64,  128, 256,  512],
                 'resnet50':  [64, 256, 512, 1024, 2048],
                 'resnet101': [64, 256, 512, 1024, 2048],
                 'eff-b0':    [16, 24,  40,  112,  1280],   # input: 224
                 'eff-b1':    [16, 24,  40,  112,  1280],   # input: 240
                 'eff-b2':    [16, 24,  48,  120,  1408],   # input: 260
                 'eff-b3':    [24, 32,  48,  136,  1536],   # input: 300
                 'eff-b4':    [24, 32,  56,  160,  1792],   # input: 380
               }

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=256,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)


        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_hrnet(cfg, is_train, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model

def get_all_indices(shape):
    indices = torch.arange(shape.numel()).view(shape)
    # indices = indices.cuda()

    out = []
    for dim in reversed(shape):
        out.append(indices % dim)
        indices = indices // dim
    return torch.stack(tuple(reversed(out)), len(shape))

class FoveaNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FoveaNet, self).__init__()
        self.cfg = cfg
        self.hrnet = get_hrnet(cfg, is_train=False, **kwargs)
        self.subpixel_up_by4 = nn.PixelShuffle(4)
        self.subpixel_up_by2 = nn.PixelShuffle(2)
        self.heatmap_ds = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0)
        )
        self.hrnet_only = cfg.TRAIN.HRNET_ONLY

        # stem net
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.bn3 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        self.orig_stemnet = cfg.TRAIN.ORIG_STEMNET
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # xf add it
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        
        # xiaofeng add for efficientnet
        if self.cfg.TRAIN.EFF_NET:
            from efficientnet.model import EfficientNet
            self.backbone_type = 'eff-b4'  # resnet34, resnet50, efficientnet-b0~b4
            self.bb_feat_dims = bb2feat_dims[self.backbone_type]
            # Set in_fpn_scheme and out_fpn_scheme to 'NA' and 'NA', respectively.
            # NA: normalize first, then add. AN: add first, then normalize.
            # self.set_fpn_layers('34', '1234', 'AN', 'AN', 'default')
            self.set_fpn_layers('34', '234', 'AN', 'AN', 'default')
            self.fpn_interp_mode = 'bilinear'  # nearest, bilinear. Using 'nearest' causes significant degradation.
            self.eff_feat_upsize = True  # Configure efficient net to generate x2 feature maps.
            self.in_fpn_use_bn = False  # If in FPN uses BN, it performs slightly worse than using GN.
            self.out_fpn_use_bn = False  # If out FPN uses BN, it performs worse than using GN.
            self.resnet_bn_to_gn = False  # Converting resnet BN to GN reduces performance.
            self.align_corners = None if self.fpn_interp_mode == 'nearest' else False
            self.G = 8
            self.cut_zeros = False
            self.output_scaleup = 1.0
            self.num_classes = 1   # for localization
            # Randomness settings
            self.hidden_dropout_prob = 0.2
            self.attention_probs_dropout_prob = 0.2
            self.out_fpn_do_dropout = False
            backbone_type = self.backbone_type.replace("eff", "efficientnet")
            stem_stride = 1 if self.eff_feat_upsize else 2
            self.backbone = EfficientNet.from_pretrained(backbone_type, advprop=True, stem_stride=stem_stride)
            print("%s created (stem stride %d)" % (backbone_type, stem_stride))
            self.backbone = EfficientNet.from_pretrained(backbone_type, advprop=True, stem_stride=stem_stride)
            if 2 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool2d((4, 4))
            elif 3 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool2d((8, 8))
            else:
                self.mask_pool = nn.AvgPool2d((16, 16))

            self.in_fpn23_conv = nn.Conv2d(self.bb_feat_dims[2], self.bb_feat_dims[3], 1)
            self.in_fpn34_conv = nn.Conv2d(self.bb_feat_dims[3], self.bb_feat_dims[4], 1)

            # in_bn4b/in_gn4b normalizes in_fpn43_conv(layer 4 features),
            # so the feature dim = dim of layer 3.
            # in_bn3b/in_gn3b normalizes in_fpn32_conv(layer 3 features),
            # so the feature dim = dim of layer 2.
            if self.in_fpn_use_bn:
                self.in_bn3b = nn.BatchNorm2d(self.bb_feat_dims[3])
                self.in_bn4b = nn.BatchNorm2d(self.bb_feat_dims[4])
                self.in_fpn_norms = [None, None, None, self.in_bn3b, self.in_bn4b]
            else:
                self.in_gn3b = nn.GroupNorm(self.G, self.bb_feat_dims[3])
                self.in_gn4b = nn.GroupNorm(self.G, self.bb_feat_dims[4])
                self.in_fpn_norms = [None, None, None, self.in_gn3b, self.in_gn4b]

            self.in_fpn_convs = [None, None, self.in_fpn23_conv, self.in_fpn34_conv]
            if self.out_fpn_layers != self.in_fpn_layers:
                self.do_out_fpn = True
                self.out_fpn12_conv = nn.Conv2d(self.bb_feat_dims[1],
                                                self.bb_feat_dims[2], 1)
                self.out_fpn23_conv = nn.Conv2d(self.bb_feat_dims[2],
                                                self.bb_feat_dims[3], 1)
                self.out_fpn34_conv = nn.Conv2d(self.bb_feat_dims[3],
                                                self.bb_feat_dims[4], 1)
                # last_out_fpn_layer = 3 here, 160 --> 1792
                last_out_fpn_layer = self.out_fpn_layers[-len(self.in_fpn_layers)]
                self.out_bridgeconv = nn.Conv2d(self.bb_feat_dims[last_out_fpn_layer],
                                                self.feat_dim, 1)
                print("out_bridgeconv (last_out_fpn_layer %d, conv: %d --> %d)" \
                      % (last_out_fpn_layer, self.bb_feat_dims[last_out_fpn_layer], self.feat_dim))

                # out_bn3b/out_gn3b normalizes out_fpn23_conv(layer 3 features),
                # so the feature dim = dim of layer 2.
                # out_bn2b/out_gn2b normalizes out_fpn12_conv(layer 2 features),
                # so the feature dim = dim of layer 1.
                if self.out_fpn_use_bn:
                    self.out_bn2b = nn.BatchNorm2d(self.bb_feat_dims[2])
                    self.out_bn3b = nn.BatchNorm2d(self.bb_feat_dims[3])
                    self.out_bn4b = nn.BatchNorm2d(self.bb_feat_dims[4])
                    self.out_fpn_norms = [None, None, self.out_bn2b, self.out_bn3b, self.out_bn4b]
                else:
                    self.out_gn2b = nn.GroupNorm(self.G, self.bb_feat_dims[2])
                    self.out_gn3b = nn.GroupNorm(self.G, self.bb_feat_dims[3])
                    self.out_gn4b = nn.GroupNorm(self.G, self.bb_feat_dims[4])
                    self.out_fpn_norms = [None, None, self.out_gn2b, self.out_gn3b, self.out_gn4b]

                self.out_fpn_convs = [None, self.out_fpn12_conv, self.out_fpn23_conv, self.out_fpn34_conv]
                self.out_conv = nn.Conv2d(self.out_feat_dim, self.num_classes, 1)
                self.out_fpn_dropout = nn.Dropout(self.hidden_dropout_prob)

                # xiaofeng add
                self.roi_gn = nn.GroupNorm(self.G, self.bb_feat_dims[4])
            # out_fpn_layers = in_fpn_layers, no need to do fpn at the output end.
            # Output class scores directly.
            else:
                self.do_out_fpn = False
                if '2' in self.in_fpn_layers:
                    # Output resolution is 1/4 of input already. No need to do upsampling here.
                    self.out_conv = nn.Conv2d(self.feat_dim, self.num_classes, 1)
                else:
                    # Output resolution is 1/8 of input. Do upsampling to make resolution x 2
                    self.out_conv = nn.ConvTranspose2d(self.feat_dim, self.num_classes, 2, 2)

            # end: xiaofeng add for efficientnet

        # fusion layer
        self.convf = nn.Conv2d(16 + 16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)

        # heatmap layer
        self.heatmap_roi = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        # regression layer
        self.regress = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.eff_regress = nn.Sequential(
            nn.Linear(1792, 160),
            nn.BatchNorm1d(160, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(160, 16),
            nn.BatchNorm1d(16, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def init_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            # Initialize low-resolution branch
            self.hrnet.init_weights(pretrained)

            # Initialize high-resolution branch
            need_init_state_dict = {}
            pretrained_state_dict = torch.load(pretrained)
            for name, m in pretrained_state_dict.items():
                cond1 = 'conv1' in name or 'conv2' in name or 'bn1' in name or 'bn2' in name
                cond2 = 'stage' in name or 'layer' in name or 'head' in name or 'transition' in name
                if cond1 and not cond2:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    # batch:        [40, 3, 112, 112]
    # nonzero_mask: [40, 28, 28]
    def get_mask(self, batch):
        with torch.no_grad():
            avg_pooled_batch = self.mask_pool(batch.abs())
            nonzero_mask = avg_pooled_batch.sum(dim=1) > 0
        return nonzero_mask

    def set_fpn_layers(self, in_fpn_layers, out_fpn_layers,
                       in_fpn_scheme, out_fpn_scheme, config_name):
        self.in_fpn_layers = [int(layer) for layer in in_fpn_layers]
        self.out_fpn_layers = [int(layer) for layer in out_fpn_layers]
        # out_fpn_layers cannot be a subset of in_fpn_layers, like: in=234, out=34.
        # in_fpn_layers  could be a subset of out_fpn_layers, like: in=34,  out=234.
        if self.out_fpn_layers[-1] > self.in_fpn_layers[-1]:
            print("in_fpn_layers=%s is not compatible with out_fpn_layers=%s" % (self.in_fpn_layers, self.out_fpn_layers))
            exit(0)

        self.in_feat_dim = self.bb_feat_dims[self.in_fpn_layers[-1]]
        self.feat_dim = self.in_feat_dim
        self.out_feat_dim = self.in_feat_dim
        self.in_fpn_scheme = in_fpn_scheme
        self.out_fpn_scheme = out_fpn_scheme
        print("'%s' in-feat: %d, feat: %d, out-feat: %d, in-scheme: %s, out-scheme: %s" % \
              (config_name, self.in_feat_dim, self.feat_dim, self.out_feat_dim,
               self.in_fpn_scheme, self.out_fpn_scheme))

    def in_fpn_forward(self, batch_base_feats, nonzero_mask, B):
        # batch_base_feat3: [40, 256, 14, 14], batch_base_feat4: [40, 512, 7, 7]
        # batch_base_feat2: [40, 128, 28, 28]
        # nonzero_mask: if '3': [40, 14, 14]; if '2': [40, 28, 28].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.in_fpn_layers[0]]

        # curr_feat: [40, 128, 28, 28] -> [40, 256, 28, 28] -> [40, 512, 28, 28]
        #                   2                   3                    4
        for layer in self.in_fpn_layers[:-1]:
            upconv_feat = self.in_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer + 1]
            if self.in_fpn_scheme == 'AN':
                curr_feat = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                        mode=self.fpn_interp_mode,
                                                        align_corners=self.align_corners)
                curr_feat = self.in_fpn_norms[layer + 1](curr_feat)
            else:
                upconv_feat_normed = self.in_fpn_norms[layer + 1](upconv_feat)
                curr_feat = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                               mode=self.fpn_interp_mode,
                                                               align_corners=self.align_corners)

        batch_base_feat_fpn = curr_feat

        H2, W2 = batch_base_feat_fpn.shape[2:]
        # batch_base_feat_fpn:        [B, 512, 28, 28]
        # batch_base_feat_fpn_hwc:    [B, 28,  28, 512]
        batch_base_feat_fpn_hwc = batch_base_feat_fpn.permute([0, 2, 3, 1])
        # vfeat_fpn:            [B, 784, 512]
        vfeat_fpn = batch_base_feat_fpn_hwc.reshape((B, -1, self.feat_dim))
        # nonzero_mask:         [B, 28, 28]
        # vmask_fpn:            [B, 784]
        vmask_fpn = nonzero_mask.reshape((B, -1))

        return vfeat_fpn, vmask_fpn, H2, W2

    def out_fpn_forward(self, batch_base_feats, vfeat_fused, B0):
        # batch_base_feat3: [40, 256, 14, 14], batch_base_feat4: [40, 512, 7, 7]
        # batch_base_feat2: [40, 128, 28, 28]
        # nonzero_mask: if '3': [40, 14, 14]; if '2': [40, 28, 28].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.out_fpn_layers[0]]
        # Only consider the extra layers in output fpn compared with input fpn,
        # plus the last layer in input fpn.
        # If in: [3,4], out: [1,2,3,4], then out_fpn_layers=[1,2,3].
        out_fpn_layers = self.out_fpn_layers[:-len(self.in_fpn_layers)]

        # curr_feat: [2, 64, 56, 56] -> [2, 128, 56, 56] -> [2, 256, 56, 56]
        #                 1                  2                   3
        for layer in out_fpn_layers:
            upconv_feat = self.out_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer + 1]
            if self.out_fpn_scheme == 'AN':
                curr_feat = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                        mode=self.fpn_interp_mode,
                                                        align_corners=self.align_corners)
                curr_feat = self.out_fpn_norms[layer + 1](curr_feat)
            else:
                upconv_feat_normed = self.out_fpn_norms[layer + 1](upconv_feat)
                curr_feat = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                               mode=self.fpn_interp_mode,
                                                               align_corners=self.align_corners)

        # curr_feat:   [2, 512, 56, 56]
        # vfeat_fused: [2, 512, 28, 28]
        out_feat_fpn = self.out_bridgeconv(curr_feat) + F.interpolate(vfeat_fused, size=curr_feat.shape[2:],
                                                                      mode=self.fpn_interp_mode,
                                                                      align_corners=self.align_corners)

        if self.out_fpn_do_dropout:
            out_feat_drop = self.out_fpn_dropout(out_feat_fpn)
            return out_feat_drop
        else:
            return out_feat_fpn

    def xf_out_fpn_heatmap(self, batch_base_feats):
        # batch_base_feat3: [B, 160, 14, 14], batch_base_feat4: [B, 1792, 7, 7]
        # batch_base_feat2: [B, 56, 28, 28], feat1: [B, 32, 56, 56]
        # nonzero_mask: if '3': [40, 14, 14]; if '2': [40, 28, 28].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.out_fpn_layers[0]]
        # Only consider the extra layers in output fpn compared with input fpn,
        # plus the last layer in input fpn.
        # If in: [3,4], out: [1,2,3,4], then out_fpn_layers=[1,2].
        out_fpn_layers = self.out_fpn_layers[:-len(self.in_fpn_layers)]
        # xf: feat0_pool:[4, 24, 224, 224] [4, 32, 112, 112] [4, 56, 56, 56] [4, 160, 28, 28] [4, 1792, 14, 14])
        # curr_feat:                       [2, 64, 56, 56]/[4, 32, 112, 112] -> [2, 128, 56, 56] -> [2, 256, 56, 56]
        #                                                      1                  2                   3
        for layer in out_fpn_layers:
            upconv_feat = self.out_fpn_convs[layer](curr_feat)   # [4, 56, 112, 112] / [4, 160, 112, 112]
            higher_feat = batch_base_feats[layer + 1]            # [4, 56, 56, 56] / [4, 160, 28, 28]
            if self.out_fpn_scheme == 'AN':
                curr_feat = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                        mode=self.fpn_interp_mode,
                                                        align_corners=self.align_corners)
                curr_feat = self.out_fpn_norms[layer + 1](curr_feat)
            else:
                upconv_feat_normed = self.out_fpn_norms[layer + 1](upconv_feat)
                curr_feat = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                               mode=self.fpn_interp_mode,
                                                               align_corners=self.align_corners)

        # curr_feat:            [B, 1792, 56, 56]   / XF: [4, 160, 112, 112]
        # batch_base_feats[-1]: [B, 1792, 7, 7]     / XF: [4, 1792, 112, 112]
        out_feat_fpn = self.out_bridgeconv(curr_feat) + F.interpolate(batch_base_feats[-1], size=curr_feat.shape[2:],
                                                                      mode=self.fpn_interp_mode,
                                                                      align_corners=self.align_corners)

        if self.out_fpn_do_dropout:
            out_feat_drop = self.out_fpn_dropout(out_feat_fpn)
            return out_feat_drop
        else:
            return out_feat_fpn


    def forward(self, input, meta, input_roi=None):
        infer_roi = input_roi is None
        ds_factor = self.cfg.MODEL.DS_FACTOR

        # Low-resolution branch
        _, _, ih, iw = input.size()   # train input is 256x256
        nh = int(ih * 1.0 / ds_factor)
        nw = int(iw * 1.0 / ds_factor)
        input_ds = F.upsample(input, size=(nh, nw), mode='bilinear', align_corners=True)
        input_ds_feats = self.hrnet(input_ds)  # (batch, 256, 64, 64)
        # 256 channel --> 16 channel, HRNET resolution: (batch, 256, 64, 64) --> (20, 16, 256, 256)
        input_ds_feats = self.subpixel_up_by4(input_ds_feats)
        # 16 channel --> 1 channel / (batch, 16, 256, 256) --> (batch, 1, 256, 256)
        heatmap_ds_pred = self.heatmap_ds(input_ds_feats)

        if self.hrnet_only:
            # Fill in the dummy data
            region_size = 2 * self.cfg.MODEL.REGION_RADIUS
            B = heatmap_ds_pred.shape[0]
            heatmap_roi_pred = torch.FloatTensor(np.zeros((B, 1, region_size, region_size), dtype=np.float32))
            offset_in_roi_pred = torch.FloatTensor(np.tile(np.array([-1, -1], np.float32), (4, 1)))
            meta.update(
                {'roi_center': offset_in_roi_pred.cpu(),
                 'input_roi': heatmap_roi_pred.cpu()
                 })
        else:
            # High-resolution branch
            region_size = 2 * self.cfg.MODEL.REGION_RADIUS
            if infer_roi:
                # Get the predicted ROI
                roi_center = get_max_preds(heatmap_ds_pred.cpu().numpy())[0][:, 0, :]
                roi_center = torch.FloatTensor(roi_center)
                roi_center *= ds_factor
                roi_center = roi_center.cuda(non_blocking=True)
                input_roi = crop_and_resize(input, roi_center, region_size)
                meta.update(
                    {'roi_center': roi_center.cpu(),
                     'input_roi': input_roi.cpu()
                     })
            else:
                assert 'roi_center' in meta.keys()
                roi_center = meta['roi_center'].cuda(non_blocking=True)

            if self.cfg.TRAIN.EFF_NET:
                # TODO -- xiaofeng change for efficient Net
                # resize to 3x224x224, changre REGION_RADIUS from 128 to 112
                # batch = cv2.resize(input_roi, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                batch = input_roi
                MOD = 0
                B0 = batch.shape[0]
                B, C, H, W = batch.shape

                with torch.no_grad():
                    # nonzero_mask: if '3': [40, 14, 14]; if '2': [40, 28, 28].
                    nonzero_mask = self.get_mask(batch)

                if self.backbone_type.startswith('resnet'):
                    batch_base_feats = self.backbone.ext_features(batch)
                elif self.backbone_type.startswith('eff'):
                    feats_dict = self.backbone.extract_endpoints(batch)
                    #                       [40, 16, 128, 128],        [40, 24, 64, 64]
                    batch_base_feats = (feats_dict['reduction_1'], feats_dict['reduction_2'], \
                                        #                       [40, 40, 32, 32],          [40, 112, 16, 16],       [40, 1280, 8, 8]
                                        feats_dict['reduction_3'], feats_dict['reduction_4'], feats_dict['reduction_5'])


                # vfeat_fpn: [B (B0*MOD), 3920, 256] / vfeat_fpn.shape = torch.Size([4, 784, 1792])
                vfeat_fpn, vmask, H2, W2 = self.in_fpn_forward(batch_base_feats, nonzero_mask, B)

                # if self.in_fpn_layers == '234', xy_shape = (28, 28)
                # if self.in_fpn_layers == '34',  xy_shape = (14, 14)
                xy_shape = torch.Size((H2, W2))
                # xy_indices: [14, 14, 20, 3]
                xy_indices = get_all_indices(xy_shape)
                scale_H = H // H2
                scale_W = W // W2

                # Has to be exactly divided.
                if (scale_H * H2 != H) or (scale_W * W2 != W):
                    import pdb
                    pdb.set_trace()

                scale = torch.tensor([[scale_H, scale_W]], device='cuda')

                # TODO
                vfeat_fused_fpn = self.xf_out_fpn_heatmap(batch_base_feats)
                # curr_feat:            [B, 1792, 56, 56]   / XF: [4, 160, 112, 112]
                # batch_base_feats[-1]: [B, 1792, 7, 7]     / XF: [4, 1792, 112, 112]

                # curr_feat:            XF: 3: [4, 160, 28, 28]
                # batch_base_feats[-1]: [B, 1792, 7, 7]     / XF: [4, 1792, 112, 112]
                # curr_feat = batch_base_feats[3]
                # vfeat_fused_fpn = self.out_bridgeconv(curr_feat) + F.interpolate(batch_base_feats[-1],
                #                                                               size=curr_feat.shape[2:],
                #                                                               mode=self.fpn_interp_mode,
                #                                                               align_corners=self.align_corners)

                # 1792 channel [B, 1792, 112, 112] --> 1 channel [B, 1, 112, 112]
                trans_scores_small = self.out_conv(vfeat_fused_fpn)

                out_size = [int(H * self.output_scaleup), int(W * self.output_scaleup)]
                # full_scores: [B0, 2, 112, 112] / XF: trans_scores_up = [B, 1, 224, 224]
                trans_scores_up = F.interpolate(trans_scores_small, size=out_size,
                                                mode='bilinear', align_corners=False)
                # import pdb
                # pdb.set_trace()
                heatmap_roi_pred = trans_scores_up
                # xiaofeng: try to generate feat for regression task for fun
                roi_feats = self.relu(self.roi_gn(vfeat_fused_fpn))

            else:
                # (batch, 16, 256, 256)
                # 3 channel --> 64 channel (batch, 3, 256, 256) --> (batch, 64, 128, 128)
                if self.cfg.TRAIN.ROI_CLAHE:
                    B, C, H, W = input_roi.shape
                    # xiaofeng add for test
                    # import pdb
                    # pdb.set_trace()
                    data_numpy = copy.deepcopy(input_roi)
                    data_numpy = data_numpy.cpu().permute([0, 2, 3, 1])
                    for i in range(B):
                        img = data_numpy.numpy()[i,:,:,:].astype('uint8')
                        # print("clahe: ", img.shape)
                        b, g, r = cv2.split(img)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                        b = clahe.apply(b)
                        g = clahe.apply(g)
                        r = clahe.apply(r)
                        img = cv2.merge([b, g, r])
                        data_numpy[i,:,:,:] = torch.from_numpy(img)
                        # print("after clahe: ", img.shape)
                    data_numpy = data_numpy.cuda().permute([0, 3, 1, 2])

                    if self.cfg.TRAIN.MV_IDEA:
                        if self.orig_stemnet:
                            roi_feats_hr = self.relu(self.bn1(self.conv1(data_numpy)))
                        else:
                            roi_feats_hr = self.relu(self.bn1(self.conv1_1(data_numpy)))
                    else:
                        roi_feats_hr = self.relu(self.bn1(self.conv1(data_numpy)))
                else:
                    if self.cfg.TRAIN.MV_IDEA:
                        if self.orig_stemnet:
                            roi_feats_hr = self.relu(self.bn1(self.conv1(input_roi)))
                        else:
                            roi_feats_hr = self.relu(self.bn1(self.conv1_1(input_roi)))
                    else:
                        roi_feats_hr = self.relu(self.bn1(self.conv1(input_roi)))     # strip = 2


                roi_feats_hr = self.relu(self.bn2(self.conv2(roi_feats_hr)))  # (batch, 64, 128, 128)
                # xiaofeng add for k=3 layer for ROI
                if self.cfg.TRAIN.MV_IDEA:
                    if not self.orig_stemnet:
                        roi_feats_hr = self.relu(self.bn3(self.conv3(roi_feats_hr)))  # (batch, 64, 128, 128)

                roi_feats_hr = self.subpixel_up_by2(roi_feats_hr)             # (batch, 16, 256, 256)

                # 256 x (res/4) channel --> 16 x channel --> (batch, 16, 256, 256)
                roi_feats_lr = crop_and_resize(input_ds_feats, roi_center/ds_factor, region_size, scale=1./ds_factor)

                # 16 + 16 channel --> (batch, 32, 256, 256) --> (336, 336)
                roi_feats = torch.cat([roi_feats_lr, roi_feats_hr], dim=1)
                roi_feats = self.relu(self.bnf(self.convf(roi_feats)))
                # 32 channel --> 1 channel  (batch, 1, 256, 256)
                heatmap_roi_pred = self.heatmap_roi(roi_feats)

            if infer_roi:
                # Get initial location from heatmap
                if self.cfg.TRAIN.MV_IDEA:
                    loc_pred_init = get_heatmap_center_preds(heatmap_roi_pred.cpu().numpy())[:, 0, :]
                else:
                    loc_pred_init = get_max_preds(heatmap_roi_pred.cpu().numpy())[0][:, 0, :]

                loc_pred_init = torch.FloatTensor(loc_pred_init).cuda(non_blocking=True)
                meta.update({'pixel_in_roi': loc_pred_init.cpu()})
            else:
                loc_pred_init = meta['pixel_in_roi'].cuda(non_blocking=True)

            # roi_feats: (batch, 32, 256, 256) --> (batch, 32, 1, 1)
            loc_init_feat = crop_and_resize(roi_feats, loc_pred_init, output_size=1)
            # (batch, 32, 1, 1) --> (batch, 32) / [B, 1792, 1, 1] --> [B, 1792]
            loc_init_feat = loc_init_feat[:, :, 0, 0]
            if self.cfg.TRAIN.EFF_NET:
                # (batch, 1792) --> (batch, 2)
                offset_in_roi_pred = self.eff_regress(loc_init_feat)
            else:
                # (batch, 32) --> (batch, 2)
                offset_in_roi_pred = self.regress(loc_init_feat)

        return heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta

def get_fovea_net(cfg, is_train, **kwargs):
    model = FoveaNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
