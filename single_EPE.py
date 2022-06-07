'''
Author: Qing Hong
Date: 2022-06-01 02:56:19
LastEditors: QingHong
LastEditTime: 2022-06-07 13:28:16
Description: file content
'''
from EPE import flow_kitti_mask_error
import os,sys
import imageio
import numpy as np
from tqdm import tqdm

def single_epe_cal(source_file_path,gt_file_path,mask_file_path):
    source_file = sorted(list(filter(lambda x:x[0]!='.',os.listdir(source_file_path))))
    gt_file  = sorted(list(filter(lambda x:x[0]!='.',os.listdir(gt_file_path))))
    mask_file = None if not mask_file_path else sorted(list(filter(lambda x:x[0]!='.',os.listdir(mask_file_path))))
    total_epe = []
    total_acc = []
    for i in tqdm(range(len(source_file))):
        source_ = os.path.join(source_file_path,source_file[i])
        gt_ = os.path.join(gt_file_path,gt_file[i])
        source = imageio.imread(source_)
        source[...,0] *= source.shape[1]
        source[...,1] *= -source.shape[0]
        gt = imageio.imread(gt_)
        gt[...,0] *= gt.shape[1]
        gt[...,1] *= -gt.shape[0]
        if mask_file:
            mask_ = os.path.join(mask_file_path,mask_file[i])
            mask = imageio.imread(mask_)[...,0]
        else:
            mask = np.ones((source.shape[0],source.shape[1]))
        epe = flow_kitti_mask_error(source[...,0],source[...,1],mask,gt[...,0],gt[...,1],mask)
        total_epe.append(epe[0])
        total_acc.append(epe[1])
    print("mean epe:{:.3f}, mean accuracy:{:.3f}%".format(np.array(total_epe).mean(),np.array(total_acc).mean()*100))
    return total_epe,total_acc

# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_low'
# mean epe:49.289, mean accuracy:20.567% times:107.76
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_medium'
# mean epe:49.095, mean accuracy:20.642% times:179.53
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_high'
# mean epe:47.803, mean accuracy:21.401% times:282.20
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_veryhigh'
# mean epe:29.149, mean accuracy:22.031% times:339.28

## source
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_left_original_mv0'
# mean epe:4.796, mean accuracy:68.750% times:289.35  mean epe:3.252, mean accuracy:86.734%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_left_original_mv0'
# mean epe:2.224, mean accuracy:83.170% times:2287.24 mean epe:3.032, mean accuracy:89.008%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_left_original_mv0'
# mean epe:8.459, mean accuracy:61.056% times:387.99  mean epe:3.875, mean accuracy:81.222%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_left_original_mv0'
# mean epe:4.406, mean accuracy:76.594% times:235.61  mean epe:3.981, mean accuracy:85.945%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_left_original_mv0'
# mean epe:4.462, mean accuracy:70.988% times:235.14  mean epe:4.565, mean accuracy:72.287%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_left_original_mv0'
# mean epe:3.814, mean accuracy:72.811% times:613.14  mean epe:4.943, mean accuracy:71.212%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_left_original_mv0'
# mean epe:1.274, mean accuracy:93.654% 1153.8s

##CF with full
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_Char_CFFULL_mv0'
# mean epe:15.551, mean accuracy:43.260% 306.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_Char_CFFULL_mv0'
# mean epe:3.032, mean accuracy:89.008% 774.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_Char_CFFULL_mv0'
# mean epe:3.992, mean accuracy:81.303% 1079.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_Char_CFFULL_mv0'
# mean epe:14.058, mean accuracy:67.757% 91.5s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_Char_CFFULL_mv0'
# mean epe:6.208, mean accuracy:52.960% 67.0s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_Char_CFFULL_mv0'
# mean epe:7.748, mean accuracy:45.492% 56.27s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_Char_CFFULL_mv0'
# mean epe:6.089, mean accuracy:82.206% 1127.4s

##Char and Bg
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_Char_mv0'
# mean epe:3.284, mean accuracy:87.212% cost times:28.32s, speed: 5.30
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_Char_mv0'
# mean epe:3.446, mean accuracy:85.850% cost times:343.97s, speed: 0.44
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_Char_mv0'
# mean epe:3.594, mean accuracy:82.856% cost times:333.15s, speed: 0.45
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_Char_mv0'
# mean epe:4.171, mean accuracy:81.794% cost times:8.45s, speed: 17.75
source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_Char_mv0'
# mean epe:3.290, mean accuracy:83.105% cost times:6.82s, speed: 21.99
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_Char_mv0'
# mean epe:5.054, mean accuracy:71.783% cost times:60.35s, speed: 2.49
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_Char_mv0'
# mean epe:2.881, mean accuracy:86.988% cost times:3850.64s, speed: 0.04

source_files = []


gt_file_path = '/Users/qhong/Desktop/inpu/scene01/mv0'
mask_file_path = '/Users/qhong/Desktop/inpu/scene01/mask'

total_epe,total_acc = single_epe_cal(source_file_path,gt_file_path,mask_file_path)

