import os
import numpy as np
import h5py as h5
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import pdb
#from xgboost import XGBClassifier
import csv
#import getDataVI
from PIL import Image
import sys
import cv2
import re
import math


def generate_AGE_prediction_from_table(filename, submit_file):
    gt_file = filename
    # all images are 2130x998 --> 2128 x 998
    # ASOCT_Name	Left_Label	X1	Y1	Right_Label	X2	Y2
    # ImageName	Fovea_X	Fovea_Y offset -->  ASOCT_NAME	X_LEFT	Y_LEFT	X_RIGHT	Y_RIGHT

    pd_db = pd.read_csv(gt_file)
    name_list = pd_db['ImageName']
    Fovea_X_list = pd_db['Fovea_X']
    Fovea_Y_list = pd_db['Fovea_Y']

    '''
    thisdict = {
          ASOCT_NAME:{"X_LEFT":X_LEFT,	"Y_LEFT":Y_LEFT, "X_RIGHT":X_RIGHT, "Y_RIGHT":Y_RIGHT },
        }
    '''
    AGE_dict = {}
    file_count = len(name_list)

    for i in range(file_count):
        filename = name_list[i]
        ASOCT_NAME = filename[2:]
        left_right = filename[0]
        Fovea_X = Fovea_X_list[i]
        Fovea_Y = Fovea_Y_list[i]
        # Wait to do, hard-code offset=1064 (=2128/2)
        # offset_list = pd_db['offset']
        if ASOCT_NAME not in AGE_dict.keys():
            AGE_dict[ASOCT_NAME] = {"X_LEFT":0,	"Y_LEFT":0, "X_RIGHT":0, "Y_RIGHT":0}
        if left_right == 'L':
            AGE_dict[ASOCT_NAME]['X_LEFT'] = round(Fovea_X)
            AGE_dict[ASOCT_NAME]['Y_LEFT'] = round(Fovea_Y)
        elif left_right == 'R':
            AGE_dict[ASOCT_NAME]['X_RIGHT'] = round(Fovea_X + 1064)
            AGE_dict[ASOCT_NAME]['Y_RIGHT'] = round(Fovea_Y)
        else:
            raise("wrong filename %s!!!" %filename)

    # TODO -- generate submission csv
    C = open(submit_file, 'w')
    C.write('ASOCT_NAME,X_LEFT,Y_LEFT,X_RIGHT,Y_RIGHT\n')

    list = []
    for key in AGE_dict.keys():
        list.append(key)

    rec_len = len(list)
    for i in range(rec_len):
        key = list[i]
        C.write('{},{},{},{},{}\n'.format(key, AGE_dict[key]['X_LEFT'], AGE_dict[key]['Y_LEFT'], \
                                       AGE_dict[key]['X_RIGHT'], AGE_dict[key]['Y_RIGHT']))



def search_coodination_from_table(filename, type="train"):
    loc_x1 = loc_y1 = loc_x2 = loc_y2 = 0
    if type == "train":
        gt_file = 'Training100/Training100_Location.xlsx'
    else:
        gt_file = 'Training100/Testing100_Location_dummy.xlsx'
    # ASOCT_Name	Left_Label	X1	Y1	Right_Label	X2	Y2

    pd_db = pd.read_excel(gt_file)
    name_list = pd_db['ASOCT_Name']

    left_X = pd_db['X1']
    left_Y = pd_db['Y1']
    right_X = pd_db['X2']
    right_Y = pd_db['Y2']

    gt_table_len = len(name_list)

    b_match = False
    for i in range(gt_table_len):
        if filename.strip().lower() == name_list[i].strip().lower():
            loc_x1 = left_X[i]
            loc_y1 = left_Y[i]
            loc_x2 = right_X[i]
            loc_y2 = right_Y[i]
            b_match = True
            break

    if not b_match:
        raise("wrong filename!!!")

    return (loc_x1, loc_y1), (loc_x2, loc_y2)


def iterate_whole_folder(rootdir='./', dir_name='train'):
    file_list = []

    file_L = []
    x_L = []
    y_L = []
    xL_offset = []
    file_R = []
    x_R = []
    y_R = []
    xR_offset = []

    data_dir = 'data/'
    data_type_dir = dir_name
    csv_save_name = 'Fovea_locations_AGE.csv'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path_with_file = os.path.join(subdir, file)
            if path_with_file.startswith('.'):
                continue
            if path_with_file.endswith('.jpg') or path_with_file.endswith('.JPG'):
                file_list.append(file)
                # use cv2 read the jpg, crop it to x4 size
                img = cv2.imread(path_with_file, cv2.IMREAD_COLOR)

                # divide it half and generate 2 filename, 2 (x, y), 2 offset list
                # 1200 images --> 2400 images
                ih, iw, ic = img.shape
                crop_iw = (iw//4)*4

                img_L = img[:, :crop_iw//2, :]
                img_R = img[:, crop_iw//2:crop_iw, :]

                new_file_L = 'L_' + file
                new_file_R = 'R_' + file

                # TODO -- we don't do it to save time
                # tmp_path_file = os.path.join(data_dir, data_type_dir, new_file_L)
                # cv2.imwrite(tmp_path_file, img_L)
                #
                # tmp_path_file = os.path.join(data_dir, data_type_dir, new_file_R)
                # cv2.imwrite(tmp_path_file, img_R)

                if dir_name == 'train':
                    # TODO -- get the original coordination first
                    (loc_x1, loc_y1), (loc_x2, loc_y2) = search_coodination_from_table(file)
                else:
                    # it is for dummy test dataset
                    (loc_x1, loc_y1), (loc_x2, loc_y2) = search_coodination_from_table(file, type="test")

                file_L.append(new_file_L)
                x_L.append(loc_x1)
                y_L.append(loc_y1)
                xL_offset.append(0)
                file_R.append(new_file_R)
                x_R.append(loc_x2-crop_iw//2)
                y_R.append(loc_y2)
                xR_offset.append(crop_iw//2)

    # save all the info to csv
    # ImageName	Fovea_X	Fovea_Y offset
    # TODO -- save the prediction into csv
    # ID_test label_test predict_test
    tmp_file = data_type_dir + '_' + csv_save_name
    tmp_path_file = os.path.join(data_dir, tmp_file)
    C = open(tmp_path_file, 'w')
    C.write(
        'ImageName,Fovea_X,Fovea_Y,offset\n')

    rec_len = len(file_L)
    for i in range(rec_len):
        C.write('{},{},{},{}\n'.format(file_L[i], x_L[i], y_L[i], xL_offset[i]))
    rec_len_1 = len(file_R)
    assert(rec_len == rec_len_1)
    for i in range(rec_len):
        C.write('{},{},{},{}\n'.format(file_R[i], x_R[i], y_R[i], xR_offset[i]))


    print("Process %s folder total [%d] images" %(rootdir, len(file_list)))
    return file_list



if __name__ == "__main__":
    type = 2
    if type == 1:
        # path = 'Training100/ASOCT_Image/'
        # iterate_whole_folder(rootdir=path, dir_name='train')
        path = 'Validation_ASOCT_Image/'
        iterate_whole_folder(rootdir=path, dir_name='test')

    else:
        # generate submission file
        data_save_dir = 'data'
        csv_save_name = 'fovea_location_results.csv'
        tmp_path_file = os.path.join(data_save_dir, csv_save_name)
        generate_AGE_prediction_from_table(tmp_path_file, 'data/Localization_Results.csv')

    print("end of program ...")