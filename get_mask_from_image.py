'''
Author: Qing Hong
Date: 2022-06-06 13:50:08
LastEditors: QingHong
LastEditTime: 2022-06-06 16:02:43
Description: file content
'''
import os,sys
import imageio
from tqdm import tqdm
from myutil import *
inp = '/Users/qhong/Desktop/inpu/scene01/mv1data'
output = '/Users/qhong/Desktop/inpu/scene01/mask'
mkdir(output)
mvs = sorted(list(filter(lambda x:x[0]!='.',os.listdir(inp))))
for i in tqdm(range(len(mvs))):
    loaded = os.path.join(inp,mvs[i])
    data = imageio.imread(loaded)
    data[:,:,0] = data[:,:,2]
    data[:,:,1] = data[:,:,2]
    data[:,:,3] = data[:,:,2]
    imageio.imwrite(os.path.join(output,'mask_{:0>6}.exr'.format(i)),data.astype("float32"))
