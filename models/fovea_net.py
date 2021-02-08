
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

class ResNet18(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        # out = torch.sigmoid(self.fc(features))
        # out1 = torch.softmax(self.fc(features), dim=1)
        return features
        # return features, out, out1

"""
ResNet34 + U-Net
"""
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_Unet(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            # disable layer0 pool to keep size same
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decode
        self.conv_decode4 = expansive_block(1024+512, 512, 512)
        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    def forward(self, x):
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        final_layer = self.final_layer(decode_block0)

        return final_layer


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
        # xiaofeng: add to replace stemNet -- test only
        # self.resnet = ResNet18()
        # TODO: increase feature channel
        self.feature_ch = 256
        self.Resnet34_Unet = Resnet34_Unet(in_channel=3, out_channel=self.feature_ch, pretrained=True)
        # self.subpixel_up_by8 = nn.PixelShuffle(8)
        self.subpixel_up_by4 = nn.PixelShuffle(4)
        self.subpixel_up_by2 = nn.PixelShuffle(2)
        self.heatmap_ds = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0)
        )
        self.hrnet_only = cfg.TRAIN.HRNET_ONLY
        self.add_heatmap_channel = False
        self.image_channel = 3
        if self.add_heatmap_channel:
            self.image_channel += 1
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
        self.conv1 = nn.Conv2d(self.image_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        # xf add it
        # if self.cfg.TRAIN.MV_IDEA:
        #     self.conv1_1 = nn.Conv2d(self.image_channel, 64, kernel_size=5, stride=1, padding=2, bias=False)
        #     self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        #     self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        # fusion layer
        self.convf = nn.Conv2d(self.feature_ch*2, self.feature_ch*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf = nn.BatchNorm2d(self.feature_ch*2, momentum=BN_MOMENTUM)
        self.convf_roi = nn.Conv2d(self.feature_ch, self.feature_ch*2, kernel_size=3, stride=1, padding=1, bias=False)

        # heatmap layer
        # self.heatmap_roi = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        # )
        self.heatmap_roi = nn.Sequential(
            # 512->128
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        # regression layer
        # self.regress = nn.Sequential(
        #     nn.Linear(32, 32),
        #     nn.BatchNorm1d(32, momentum=BN_MOMENTUM),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 16),
        #     nn.BatchNorm1d(16, momentum=BN_MOMENTUM),
        #     nn.ReLU(),
        #     nn.Linear(16, 2)
        # )
        # if self.cfg.TRAIN.EFF_NET:
        #     self.eff_regress = nn.Sequential(
        #         nn.Linear(1792, 160),
        #         nn.BatchNorm1d(160, momentum=BN_MOMENTUM),
        #         nn.ReLU(),
        #         nn.Dropout(0.5),
        #         nn.Linear(160, 16),
        #         nn.BatchNorm1d(16, momentum=BN_MOMENTUM),
        #         nn.ReLU(),
        #         nn.Linear(16, 2)
        # )

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

    def forward(self, input, meta, input_roi=None):
        remove_hr_feature = True
        infer_roi = input_roi is None
        ds_factor = self.cfg.MODEL.DS_FACTOR

        # Low-resolution branch
        batch, _, ih, iw = input.size()   # train input is 256x256
        nh = int(ih * 1.0 / ds_factor)
        nw = int(iw * 1.0 / ds_factor)
        input_ds = F.upsample(input, size=(nh, nw), mode='bilinear', align_corners=True)
        input_ds_feats_orig = self.hrnet(input_ds)  # (batch, 256, 64, 64)
        # 256 channel --> 16 channel, HRNET resolution: (batch, 256, 64, 64) --> (20, 16, 256, 256)
        input_ds_feats = self.subpixel_up_by4(input_ds_feats_orig)
        # 16 channel --> 1 channel / (batch, 16, 256, 256) --> (batch, 1, 256, 256)
        heatmap_ds_pred = self.heatmap_ds(input_ds_feats)

        # High-resolution branch
        region_size = 2 * self.cfg.MODEL.REGION_RADIUS
        if infer_roi:
            # Get the predicted ROI
            roi_center = get_max_preds(heatmap_ds_pred.cpu().numpy())[0][:, 0, :]
            roi_center = torch.FloatTensor(roi_center)
            roi_center *= ds_factor
            roi_center = roi_center.cuda(non_blocking=True)
            input_roi = crop_and_resize(input, roi_center, region_size)
            # TODO: we could generate 3 different ROI here
            meta.update(
                {'roi_center': roi_center.cpu(),
                 'input_roi': input_roi.cpu()
                 })
        else:
            assert 'roi_center' in meta.keys()
            roi_center = meta['roi_center'].cuda(non_blocking=True)

        if self.add_heatmap_channel:
            heatmap_2_roilayer = crop_and_resize(heatmap_ds_pred, roi_center/ds_factor, region_size, scale=1. / ds_factor)
            # TODO: add to input_roi
            input_roi = torch.stack([input_roi, heatmap_2_roilayer], dim=1)

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
            roi_feats_hr = self.relu(self.bn2(self.conv2(roi_feats_hr)))  # (batch, 64, 128, 128)
        else:
            if self.cfg.TRAIN.MV_IDEA:
                if self.orig_stemnet:
                    roi_feats_hr = self.relu(self.bn1(self.conv1(input_roi)))
                else:
                    roi_feats_hr = self.relu(self.bn1(self.conv1_1(input_roi)))
                roi_feats_hr = self.relu(self.bn2(self.conv2(roi_feats_hr)))  # (batch, 64, 128, 128)
            else:
                if self.orig_stemnet:
                    roi_feats_hr = self.relu(self.bn1(self.conv1(input_roi)))     # strip = 2
                    roi_feats_hr = self.relu(self.bn2(self.conv2(roi_feats_hr)))  # (batch, 64, 128, 128)
                else:
                    # roi_feats_hr = self.resnet(input_roi)    # (batch, 512, 8, 8)
                    # don't apply Resnet34_Unet if remove_hr_feature
                    if not remove_hr_feature:
                        roi_feats_hr = self.Resnet34_Unet(input_roi)  # (batch, 16, 128, 128)
                        roi_feats_hr = F.interpolate(roi_feats_hr, region_size, mode="bilinear") # (batch, 16, 256, 256)

            # xiaofeng add for k=3 layer for ROI
            if self.cfg.TRAIN.MV_IDEA:
                if not self.orig_stemnet:
                    roi_feats_hr = self.relu(self.bn3(self.conv3(roi_feats_hr)))  # (batch, 64, 128, 128)

            if self.orig_stemnet:
                roi_feats_hr = self.subpixel_up_by2(roi_feats_hr)             # (batch, 16, 256, 256)
            else:
                # Note: this is to test the stemNet replacing performance
                # for resnet only
                # roi_feats_hr = self.subpixel_up_by4(roi_feats_hr)             # (batch, 16, 32, 32)
                # TODO: for Resnet34_Unet
                # roi_feats_hr = crop_and_resize(roi_feats_hr, roi_center, region_size, scale=1.0)
                pass

                # 256 x (res/4) channel --> 16 x channel --> (batch, 16, 256, 256)
            # crop_and_resize(image, center, output_size, scale=1)

            # this is the original operation
            # roi_feats_lr = crop_and_resize(input_ds_feats, roi_center/ds_factor, region_size, scale=1./ds_factor)

            # TODO: xiaofeng use 256 channel : 64x64 -> 256x256
            roi_feats_lr = crop_and_resize(input_ds_feats_orig, roi_center / ds_factor, region_size, scale=1.0)


            # 16 + 16 channel --> (batch, 32, 256, 256) --> (336, 336)
            # fusion layer
            if remove_hr_feature:
                # discard ROI feature to verify refine stage performance, we don't concat the feature
                roi_feats = self.relu(self.bnf(self.convf_roi(roi_feats_lr)))
                # end of discard ROI feature
            else:
                roi_feats = torch.cat([roi_feats_lr, roi_feats_hr], dim=1)  # (batch, 512/32, 256, 256)
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
            # loc_init_feat = crop_and_resize(roi_feats, loc_pred_init, output_size=1)
            # (batch, 32, 1, 1) --> (batch, 32) / [B, 1792, 1, 1] --> [B, 1792]
            # loc_init_feat = loc_init_feat[:, :, 0, 0]

            # xiaofeng: disable regression network
            offset_in_roi_pred = torch.from_numpy(np.tile(np.array([-1, -1], np.float32), (batch, 1))).cuda()
            # if self.cfg.TRAIN.EFF_NET:
            #     # (batch, 1792) --> (batch, 2)
            #     offset_in_roi_pred = self.eff_regress(loc_init_feat)
            # else:
            #     # (batch, 32) --> (batch, 2)
            #     offset_in_roi_pred = self.regress(loc_init_feat)

        return heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta

def get_fovea_net(cfg, is_train, **kwargs):
    model = FoveaNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
