'''
Author: Qing Hong
Date: 2022-06-01 01:40:32
LastEditors: QingHong
LastEditTime: 2022-06-02 03:56:12
Description: file content
'''
import os,sys
import imageio,cv2
import numpy as np
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)


type = ['mv0','mv1'][0]
target = '/Users/qhong/Desktop/resout'
mask_file = '/Users/qhong/Desktop/inpu/scene01/video'
save_path = '/Users/qhong/Desktop/trans_output'
mkdir(save_path)
inp = sorted(list(filter(lambda x:x[0]!='.',os.listdir(mask_file))))
target_files = list(filter(lambda x:x[0]!='.',os.listdir(target)))

for target_file in target_files:
    cur_file = os.path.join(target,target_file)
    mkdir(os.path.join(save_path,target_file))
    images = sorted(list(filter(lambda x:x[0]!='.',os.listdir(cur_file))))
    for i in range(len(images)):
        image = images[i]
        mask_image = imageio.imread(os.path.join(mask_file,inp[i]))
        cur_image = os.path.join(cur_file,image)
        image = imageio.imread(cur_image).astype('float32')
        res_image = np.zeros_like(mask_image).astype('float32')
        h,w,_ = res_image.shape
        # x0>=0&&x0<s32W&&y0>=0&&y0<s32H&&z0<cvmZBuffer0.at<float>(y0,x0)

        for i in range(h):
            for j in range(w):
                
                res_image[i,j,:2] = image[i*2,j*2,:2] - mask_image[i,j,:2]
                res_image[np.where(abs(res_image)>150)] = 0

        if type =='mv0':
            image[...,0]/= image.shape[1]
            image[...,1]/=-image.shape[0]
        else:
            image[...,0]/= -image.shape[1]
            image[...,1]/= image.shape[0]
        image = cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
        image[...,2] = mask_image[...,2]
        save= os.path.join(save_path,target_file,'mv_{:0>6}.exr'.format(i))
        imageio.imwrite(save,image.astype("float32"))

