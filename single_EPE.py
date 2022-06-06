'''
Author: Qing Hong
Date: 2022-06-01 02:56:19
LastEditors: QingHong
LastEditTime: 2022-06-06 14:02:32
Description: file content
'''
from EPE import flow_kitti_mask_error
import os,sys
import imageio
import numpy as np
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_low'
# mean epe:49.289, mean accuracy:20.567% times:107.76
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_medium'
# mean epe:49.095, mean accuracy:20.642% times:179.53
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_high'
# mean epe:47.803, mean accuracy:21.401% times:282.20
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_veryhigh'
# mean epe:29.149, mean accuracy:22.031% times:339.28

# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_left_original_mv0'
# mean epe:2.224, mean accuracy:83.170% times:2287.24
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_left_original_mv0'
# mean epe:4.796, mean accuracy:68.750% times:289.35
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_left_original_mv0'
# mean epe:8.459, mean accuracy:61.056% times:387.99
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_left_original_mv0'
# mean epe:4.406, mean accuracy:76.594% times:235.61
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_left_original_mv0'
# mean epe:4.462, mean accuracy:70.988% times:235.14
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_left_original_mv0'
# mean epe:3.814, mean accuracy:72.811% times:613.14

gt_file_path = '/Users/qhong/Desktop/inpu/scene01/mv0'
def single_epe_cal(source_file_path,gt_file_path):
    source_file = sorted(list(filter(lambda x:x[0]!='.',os.listdir(source_file_path))))
    gt_file  = sorted(list(filter(lambda x:x[0]!='.',os.listdir(gt_file_path))))
    total_epe = []
    total_acc = []
    for i in range(len(source_file)):
        source_ = os.path.join(source_file_path,source_file[i])
        gt_ = os.path.join(gt_file_path,gt_file[i])
        source = imageio.imread(source_)
        source[...,0] *= source.shape[1]
        source[...,1] *= -source.shape[0]
        gt = imageio.imread(gt_)
        gt[...,0] *= gt.shape[1]
        gt[...,1] *= -gt.shape[0]
        epe = flow_kitti_mask_error(source[...,0],source[...,1],np.ones((source.shape[0],source.shape[1])),gt[...,0],gt[...,1],np.ones((source.shape[0],source.shape[1])))
        total_epe.append(epe[0])
        total_acc.append(epe[1])
    print("mean epe:{:.3f}, mean accuracy:{:.3f}%".format(np.array(total_epe).mean(),np.array(total_acc).mean()*100))
    return total_epe,total_acc
