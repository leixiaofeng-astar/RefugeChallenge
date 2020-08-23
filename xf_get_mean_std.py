"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre

python3 xf_get_mean_std.py --dirs REFUGE-Training400/Training400,REFUGE-Validation400,REFUGE-Test400,Refuge2-Validation

python3 xf_get_mean_std.py --dirs REFUGE-Validation400,REFUGE-Test400,Refuge2-Validation

"""

import numpy as np
import sys
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import argparse
import imgaug.augmenters as iaa
import copy
parser = argparse.ArgumentParser()

parser.add_argument('--dirs', dest='img_dirs', type=str, required=True, 
                    help='Root of Image dataset(s). Can specify multiple directories (separated with ",")')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5, 
                    help='Convert images to grayscale by so much degree.')
parser.add_argument('--size', dest='chosen_size', type=int, default=0, 
                    help='Use images of this size (among all cropping sizes). Default: 0, i.e., use all sizes.')
args = parser.parse_args()
                    
# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
img_types = ['png', 'jpg']

def img_preprocess_1(img):
    # load data
    data_numpy = copy.deepcopy(img)
    if data_numpy is None:
        print('=> fail to read image')

    # xiaofeng add for test
    alpha = 0.5
    img_temp = data_numpy.copy()
    # img_gray = (img_temp[:, :, 0] + img_temp[:, :, 1] + img_temp[:, :, 2]) / 3
    img_gray = img_temp[:, :, 0] * 0.11 + img_temp[:, :, 1] * 0.59 + img_temp[:, :, 2] * 0.3
    img_gray2 = img_gray * alpha
    img_gray2 = img_gray2.reshape(img_gray2.shape[0], img_gray2.shape[1], -1)
    img_gray3 = np.tile(img_gray2, [1, 1, 3])

    data_numpy = data_numpy.astype(np.float)
    data_numpy = data_numpy * alpha + img_gray3

    # cmin = data_numpy.min()
    # cmax = data_numpy.max()
    # Thr0 = cmax - (cmax*0.02)
    # Thr0 = 255
    # if (cmax > Thr0):
    #     cmax = Thr0
    #     d2 = data_numpy[data_numpy <= Thr0]
    #     cmax2 = d2.max()
    #     data = (data_numpy.clip(0, cmax2)).astype(np.uint16)
    # else:
    #     data = (data_numpy.clip(0, cmax)).astype(np.uint16)
    #     cmax2 = cmax
    #
    # scale = float(255.0) / cmax2
    # if scale == 0:
    #     scale = 1
    # bytedata = (data - 0) * scale
    # data_numpy = (bytedata.clip(0, 255)).astype(np.uint8)

    data_numpy = (data_numpy.clip(0, 255)).astype(np.uint8)
    return data_numpy


def img_preprocess_2(img):
    # load data
    data_numpy = copy.deepcopy(img)
    # xiaofeng add for test
    b, g, r = cv2.split(data_numpy)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    data_numpy = cv2.merge([b, g, r])
    return data_numpy


def cal_dir_stat(root, gray_alpha, chosen_size):
    cls_dirs = [ d for d in listdir(root) if isdir(join(root, d)) and 'label' not in d ]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    gray_trans = iaa.Grayscale(alpha=gray_alpha)
    
    for idx, d in enumerate(cls_dirs):
        print("Class '{}'".format(d))
        im_paths = []
        for img_type in img_types:
            im_paths += glob(join(root, d, "*.{}".format(img_type)))
        if chosen_size:
            all_sizes_count = len(im_paths)
            im_paths = filter(lambda name: '_{}_'.format(chosen_size) in name, im_paths)
            im_paths = list(im_paths)
            print("{} size {} images chosen from {}".format(len(im_paths), chosen_size, all_sizes_count))
            
        for path in im_paths:
            im = cv2.imread(path) # image in M*N*CHANNEL_NUM shape, channels in BGR order
            # xiaofeng change -- the output must be R, G, B sequence
            im = im[:, :, ::-1]   # Change channels to RGB
            im = gray_trans.augment_image(im)
            # im = im[:, :, ::-1]  # Change channels to RGB
            # print("process %s " %path)
            # im = img_preprocess_1(im)

            im = im/255.0
            pixel_num += (im.size/CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))
    
    return rgb_mean, rgb_std

# The script assumes that under img_dir_path, there are separate directories for each class
# of training images.
img_dirs = args.img_dirs.split(",")
# img_dirs = ['REFUGE-Training400/Training400', \
#            'REFUGE-Validation400/REFUGE-Validation400', \
#            'REFUGE-Test400/Test400', \
#            'Refuge2-Validation/Refuge2-Validation']

for img_dir in img_dirs:
    img_dir_path = "/home/user/eye/xf_refuge/data/{}/".format(img_dir)
    print("Calculating {}...".format(img_dir_path))
    start = timeit.default_timer()
    mean, std = cal_dir_stat(img_dir_path, args.gray_alpha, args.chosen_size)
    end = timeit.default_timer()
    print("elapsed time: {}".format(end-start))
    mean_str = ", ".join([ "%.3f" %x for x in mean ])
    std_str  = ", ".join([ "%.3f" %x for x in std ])

    print("mean:\n[{}]\nstd:\n[{}]".format(mean_str, std_str))
