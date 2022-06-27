import os,sys
from tqdm import tqdm
import imageio
import numpy as np
from myutil import mkdir
import re
def restrain(flow,mask_,threshold=1):
    mask = mask_[...,0]
    if mask.max()<=1 and mask.min()>=-1:
        threshold = 1
    flow[np.where(mask<threshold)] = 0
    return flow

'''
description: 生成差异文件
param {*} root 源文件根目录（图像+mask）
param {*} res_root 结果根目录（mv结果目录）
param {*} output_root 存放目录
param {*} gt_file_name gt文件名
param {*} mask_file_name mask文件名
param {*} res_name mv文件名
return {*}
'''
def make_different(root,res_root,output_root,gt_file_name,mask_file_name,res_name):
    #load gt file and mask file
    scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))
    gt_dict = {}
    mask_dict = {}
    for scene in scenes:
        gt_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,gt_file_name)))))
        gt_dict[scene] = [os.path.join(root,scene,gt_file_name,i) for i in gt_files]
        mask_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,mask_file_name)))))
        mask_dict[scene] = [os.path.join(root,scene,mask_file_name,i) for i in mask_files]


    mkdir(output_root)
    #load result file
    res_scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(res_root))))
    for res_scene in res_scenes:
        mkdir(os.path.join(output_root,res_scene))
        tar_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(res_root,res_scene,res_name)))))
        for i in tqdm(range(len(tar_files)),desc=res_scene):
            gt = imageio.imread(gt_dict[res_scene][i])
            sr = imageio.imread(os.path.join(res_root,res_scene,res_name,tar_files[i]))
            mask = imageio.imread(mask_dict[res_scene][i])
            diff = restrain(np.abs(gt - sr),mask,1)
            save_path = os.path.join(output_root,res_scene,res_name+'_diff')
            mkdir(save_path)
            save_path += '/diff_'+re.findall(r'\d+', tar_files[i].split('/')[-1])[-1]+ '.exr'
            imageio.imwrite(save_path,diff.astype("float32"))

'''
description: 生成差异文件 使用mv0+mv1
param {*} root 源文件根目录（图像+mask）
param {*} res_root 结果根目录（mv结果目录）
param {*} output_root 存放目录
param {*} gt_file_name gt文件名
param {*} mask_file_name mask文件名
param {*} res_name_mv0 mv0文件名
param {*} res_name_mv1 mv1文件名
return {*}
'''
def make_different_binary(root,res_root,output_root,gt_file_name,mask_file_name,res_name_mv0,res_name_mv1):
    #load gt file and mask file
    scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))
    gt_dict = {}
    mask_dict = {}
    for scene in scenes:
        gt_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,gt_file_name)))))
        gt_dict[scene] = [os.path.join(root,scene,gt_file_name,i) for i in gt_files]
        mask_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,mask_file_name)))))
        mask_dict[scene] = [os.path.join(root,scene,mask_file_name,i) for i in mask_files]


    mkdir(output_root)
    #load result file
    res_scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(res_root))))
    for res_scene in res_scenes:
        mkdir(os.path.join(output_root,res_scene))
        tar_files_mv0 = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(res_root,res_scene,res_name_mv0)))))
        tar_files_mv1 = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(res_root,res_scene,res_name_mv1)))))
        for i in tqdm(range(len(tar_files_mv0)-1),desc=res_scene):
            gt = imageio.imread(gt_dict[res_scene][i])
            sr_mv0 = imageio.imread(os.path.join(res_root,res_scene,res_name_mv0,tar_files_mv0[i]))
            sr_mv1 = imageio.imread(os.path.join(res_root,res_scene,res_name_mv0,tar_files_mv1[i]))
            sr = sr_mv0/2 - sr_mv1/2
            mask = imageio.imread(mask_dict[res_scene][i])
            diff = restrain(np.abs(gt - sr),mask,1)
            save_path = os.path.join(output_root,res_scene,res_name_mv0+'_diff_binary')
            mkdir(save_path)
            save_path += '/diff_'+re.findall(r'\d+', tar_files_mv0[i].split('/')[-1])[-1]+ '.exr'
            imageio.imwrite(save_path,diff.astype("float32"))

root = '/Users/qhong/Desktop/opt_test_datasets/optical_test_pattern'
gt_file_name = 'mv0'
mask_file_name = 'mask'
res_root = '/Users/qhong/Desktop/opt_test_datasets/output'
res_name = 'pca_flow_Char_mv0'
output_root='/Users/qhong/Desktop/opt_test_datasets/output'
# make_different(root,res_root,output_root,gt_file_name,mask_file_name,res_name)
res_name_mv1 = 'pca_flow_Char_mv1'
make_different_binary(root,res_root,output_root,gt_file_name,mask_file_name,res_name,res_name_mv1)