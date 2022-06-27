'''
Author: Qing Hong
Date: 2022-06-06 13:50:08
LastEditors: QingHong
LastEditTime: 2022-06-10 17:50:34
Description: file content
'''
import os,sys
import imageio
from tqdm import tqdm
from myutil import *
import cv2
inp = '/Users/qhong/Desktop/inpu/scene01/mv1data'
output = '/Users/qhong/Desktop/inpu/scene01/mask_png'
root = '/Users/qhong/Desktop/opt_test_datasets/optical_test_pattern'

'''
description: 提取多个场景的mask
param {*} root
return {*}
'''
def get_mask_from_images(root,mv_name = 'mv1data'):
    scenes = sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))
    for scene in scenes:
        first = os.path.join(root,scene)
        inp = os.path.join(first,mv_name)
        output = os.path.join(first,'mask')
        get_mask_from_image(inp,output)

'''
description: 提取单个场景的mask
param {*} inp
param {*} output
return {*}
'''
def get_mask_from_image(inp,output):
    shutil.rmtree(output,ignore_errors=True)
    mkdir(output)
    mvs = sorted(list(filter(lambda x:x[0]!='.',os.listdir(inp))))
    for i in tqdm(range(len(mvs))):
        loaded = os.path.join(inp,mvs[i])
        data = imageio.imread(loaded)
        data[:,:,0] = data[:,:,2]
        data[:,:,1] = data[:,:,2]
        data[:,:,3] = data[:,:,2]
        # imageio.imwrite(os.path.join(output,'mask_{:0>6}.exr'.format(i)),data.astype("float32"))
        imageio.imwrite(os.path.join(output,'mask_{:0>6}.exr'.format(i)),data.astype("float32"))

def get_mask_from_image_png(inp,output):
    shutil.rmtree(output,ignore_errors=True)
    mkdir(output)
    mvs = sorted(list(filter(lambda x:x[0]!='.',os.listdir(inp))))
    for i in tqdm(range(len(mvs))):
        loaded = os.path.join(inp,mvs[i])
        data = imageio.imread(loaded)
        data[:,:,0] = data[:,:,2]
        data[:,:,1] = data[:,:,2]
        # imageio.imwrite(os.path.join(output,'mask_{:0>6}.exr'.format(i)),data.astype("float32"))
        cv2.imwrite(os.path.join(output,'mask_{:0>6}.png'.format(i)),data.astype("int32"))


get_mask_from_image_png(inp,output)