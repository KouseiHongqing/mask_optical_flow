'''
Author: Qing Hong
Date: 2022-03-08 17:21:31
LastEditors: QingHong
LastEditTime: 2022-06-06 10:41:43
Description: file content
'''
import os
import time
import shutil
import numpy as np

cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)

def build_dir(mk_dir,algo):
    if  not os.path.exists(mk_dir):
        os.makedirs(mk_dir)

    save_dir =  mk_dir + '/' + algo +'/'+ cur_time_sec
    if  os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print('saving file created:{}'.format(save_dir))

import matplotlib.pyplot as plt
def pt(image):
    plt.imshow(image[...,::-1])
def pt2(image):
    plt.imshow(np.insert(image,1,0,axis=2)[...,::-1])
def plot(image):
    plt.imshow(image)
from PIL import Image

def show(image):
    im = Image.fromarray(image.astype('uint8')).convert('RGB')
    im.show()

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)

def delpoint(root,arr):
        tmp = []
        for ar in arr:
            if ar[0] != '.' and os.path.isdir(root+'/' +ar):
                tmp.append(ar)
        return tmp

def prune_point(file):
    res = []
    for i in file:
        if i[0]!='.':
            res.append(i)
    return res

# import png
# def saveUint16(path,z):
#     # Use pypng to write zgray as a grayscale PNG.
#     with open(path, 'wb') as f:
#         writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
#         zgray2list = z.tolist()
#         writer.write(f, zgray2list)

# def depthToint16(dMap, minVal=0, maxVal=10):
#     #Maximum and minimum distance of interception 
#     dMap[dMap>maxVal] = maxVal
#     # print(np.max(dMap),np.min(dMap))
#     dMap = ((dMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
#     return dMap

# def normalizationDepth(depthfile, savepath):
#     correctDepth = readDepth(depthfile)
#     depth = depthToint16(correctDepth, 0, 10)
#     saveUint16(depth,savepath)

