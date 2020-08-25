from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import cv2
import torch.nn.functional as F

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, 1, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, 1, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, 1, 1))
    idx = idx.reshape((batch_size, 1, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # TODO: xiaofeng comment: get x, y of the hotest point
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# xiaofeng add for MV processing to get circle
def get_heatmap_center_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, 1, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    # heatmaps_reshaped = batch_heatmaps.reshape((batch_size, 1, -1))
    batch_heatmaps = batch_heatmaps.transpose((0, 2, 3, 1))
    preds = np.zeros((batch_size, 1, 2), dtype = int)

    for idx in range(batch_size):
        center_x = width // 2
        center_y = height // 2
        preds[idx, 0, 0] = center_x
        preds[idx, 0, 1] = center_y

        # handle it as one image
        batch_heatmaps[batch_heatmaps < 0] = 0
        img = np.uint8( batch_heatmaps[idx, :, :, :] * 255)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # https://blog.csdn.net/weixin_42904405/article/details/82814768
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=30)
        max_rad = 0.0
        # import pdb
        # pdb.set_trace()
        if circles is not None:
            for i in circles[0, :]:
                # get the maximum outer circle
                current_rad = i[2]
                if current_rad > max_rad:
                    max_rad = current_rad
                    preds[idx, 0, 0] = i[0]
                    preds[idx, 0, 1] = i[1]

            #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            #
            # tmp_filename = "test_roi_heatmap_%d.png" %(idx)
            # cv2.imwrite(tmp_filename, cimg)

    return preds

def get_final_preds(config, batch_heatmap_ds, batch_heatmap_roi, offsets_in_roi, meta):
    coords_ds, maxvals_ds = get_max_preds(batch_heatmap_ds)
    if config.TRAIN.MV_IDEA:
        coords_roi = get_heatmap_center_preds(batch_heatmap_roi)
    else:
        coords_roi, maxvals_roi = get_max_preds(batch_heatmap_roi)

    region_size = 2 * config.MODEL.REGION_RADIUS
    offsets_in_roi = offsets_in_roi * region_size
    # coords: [N, 1, 2] -> [N, 2]
    coords_ds = coords_ds[:, 0, :]
    coords_roi = coords_roi[:, 0, :]
    coords_lr = coords_ds * config.MODEL.DS_FACTOR
    coords_hr = coords_roi + meta['roi_center'].cpu().numpy() - config.MODEL.REGION_RADIUS
    coords_final = coords_hr + offsets_in_roi
    coords_roi_final = coords_roi + offsets_in_roi
    return coords_lr, coords_hr, coords_final, coords_roi, coords_roi_final


