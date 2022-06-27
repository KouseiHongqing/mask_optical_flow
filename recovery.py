'''
Author: Qing Hong
Date: 2022-06-15 14:17:27
LastEditors: QingHong
LastEditTime: 2022-06-15 16:22:10
Description: file content
'''
import os,sys
from pydoc import describe
from tqdm import tqdm
import imageio
import numpy as np
from myutil import mkdir,imresize
import re
import cv2
from png_to_mp4 import raw_png2mp4
def restrain(flow,mask_,threshold=1):
    mask = mask_[...,0]
    if mask.max()<=1 and mask.min()>=-1:
        threshold = 1
    flow[np.where(mask<threshold)] = 0
    return flow

'''
description: 回复文件
param {*} root 源文件根目录（图像+mask）
param {*} res_root 结果根目录（mv结果目录）
param {*} output_root 存放目录
param {*} image_name 图片文件名
param {*} mask_file_name mask文件名
param {*} res_name mv文件名
return {*}
'''
def recover_image(root,res_root,output_root,image_name,mask_file_name,res_name):
    #load gt file and mask file
    scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))
    image_dict = {}
    mask_dict = {}
    for scene in scenes:
        image_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,image_name)))))
        image_dict[scene] = [os.path.join(root,scene,image_name,i) for i in image_files]
        mask_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(root,scene,mask_file_name)))))
        mask_dict[scene] = [os.path.join(root,scene,mask_file_name,i) for i in mask_files]


    mkdir(output_root)
    #load result file
    res_scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(res_root))))
    for res_scene in res_scenes:
        mkdir(os.path.join(output_root,res_scene))
        tar_files = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(res_root,res_scene,res_name)))))
        for i in tqdm(range(len(tar_files)),desc=res_scene):
            img = cv2.imread(image_dict[res_scene][i])
            mv = imageio.imread(os.path.join(res_root,res_scene,res_name,tar_files[i]))
            mask = imageio.imread(mask_dict[res_scene][i])
            img = restrain(img,mask)
            mv[...,0] *= mv.shape[1]
            mv[...,1] *= -mv.shape[0]
            img = immc(img,mv)
            save_path = os.path.join(output_root,res_scene,res_name+'_restore')
            mkdir(save_path)
            save_path += '/img'+re.findall(r'\d+', tar_files[i].split('/')[-1])[-1]+ '.png'
            cv2.imwrite(save_path,img)

def immc(input, mv,mvoffset=None):
    indtype = input.dtype
    input = input.astype(np.float32)
    output = np.zeros_like(input, dtype=np.float32)
    h, w = input.shape[:2]
    if mvoffset is None:
        mvoffset = 0,0
    ratio = (input.shape[0] + mv.shape[0] - 1) // mv.shape[0]
    mv = mv.repeat(ratio, axis=0).repeat(ratio, axis=1)[:h, :w]

    mvoffset = mvoffset[0]*ratio, mvoffset[1]*ratio

    if mvoffset[0] != 0 or mvoffset[1] != 0:
        mv[:,:] += mvoffset

    mv_i = np.floor(mv).astype(np.intp)
    mv_f = mv - mv_i
    w0 = (1 - mv_f[...,0]) * (1 - mv_f[...,1])
    w1 = (    mv_f[...,0]) * (1 - mv_f[...,1])
    w2 = (1 - mv_f[...,0]) * (    mv_f[...,1])
    w3 = (    mv_f[...,0]) * (    mv_f[...,1])
    if input.ndim == 3:
        w0, w1, w2, w3 = w0[...,np.newaxis], w1[...,np.newaxis], w2[...,np.newaxis], w3[...,np.newaxis]
    y, x = np.ix_(np.arange(h, dtype=np.intp), np.arange(w, dtype=np.intp)) # y: (h,1), x: (1, w)
    x0 = x + mv_i[...,0]
    y0 = y + mv_i[...,1]
    x1 = (x0 + 1).clip(0, w-1)
    y1 = (y0 + 1).clip(0, h-1)
    x0 = x0.clip(0, w-1)
    y0 = y0.clip(0, h-1)
    output = input[y0, x0] * w0 + input[y0, x1] * w1 + input[y1, x0] * w2 + input[y1, x1] * w3

    return output.astype(indtype)


def make_double_mp4(left,right,save_path,llmit,rlmit,fps=10,masks=None,restrain_left=False,restrain_right=False):
    res = []
    for i in tqdm(range(len(left)-1),desc='reading image'):
        mask = None if not masks else imageio.imread(masks[i])
        l = cv2.imread(left[i])
        r = cv2.imread(right[i])
        if restrain_left:
            l = restrain(l,mask)
        if restrain_right:
            r = restrain(r,mask)
            
        conc = image_concat(l,r,llmit,rlmit)
        res.append(conc)
    raw_png2mp4(res,save_path,fps)


def image_concat(left,right,llmit,rlmit):
    left = left[:,llmit:rlmit,:]
    right = right[:,llmit:rlmit,:]
    h,w,c = left.shape
    res = np.zeros((h,w*2,c))
    res[:,:w,:] = left
    res[:,-w:,:] = right
    return res.astype('uint8')

def demo(lroot,rroot,output,lname,rname,maskname=None,restrain_left=False,restrain_right=False):
    scenes_l = sorted(list(filter(lambda x:x[0]!='.',os.listdir(lroot))))
    scenes_r = sorted(list(filter(lambda x:x[0]!='.',os.listdir(rroot))))
    l_dict = {}
    mask_dict={}
    for scene in scenes_l:
        tmps = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(lroot,scene,lname)))))
        l_dict[scene] = [os.path.join(lroot,scene,lname,i) for i in tmps]
        if maskname:
            mask_tmps = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(lroot,scene,maskname)))))
            mask_dict[scene] = [os.path.join(lroot,scene,maskname,i) for i in mask_tmps]
    for scene in scenes_r:
        left = l_dict[scene]
        tmps = sorted(list(filter(lambda x:x[0]!='.',os.listdir(os.path.join(rroot,scene,rname)))))
        right = [os.path.join(rroot,scene,rname,i) for i in tmps]
        mkdir(output)
        save_path = os.path.join(output,scene+'.mp4')
        mask = None
        if maskname:
            mask = mask_dict[scene]
        make_double_mp4(left,right,save_path,400,800,masks=mask,restrain_left=restrain_left,restrain_right=restrain_right)


root = '/Users/qhong/Desktop/opt_test_datasets/optical_test_pattern'
gt_file_name = 'mv0'
mask_file_name = 'mask'
res_root = '/Users/qhong/Desktop/opt_test_datasets/output'
res_name = 'pca_flow_Char_mv0'
output_root='/Users/qhong/Desktop/opt_test_datasets/output'
# make_different(root,res_root,output_root,gt_file_name,mask_file_name,res_name)
res_name_mv1 = 'pca_flow_Char_mv1'
image_name = 'video'
# recover_image(root,res_root,output_root,image_name,mask_file_name,res_name)
demo(root,output_root,'/Users/qhong/Desktop/opt_test_datasets/mp4output','video','pca_flow_Char_mv0_restore',maskname='mask',restrain_left=True)