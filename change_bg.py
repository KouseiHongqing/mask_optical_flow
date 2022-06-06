
import os,sys
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_dir = 'src'
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


def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)

root = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/421test'
list_seq_file = delpoint(root,sorted(os.listdir(root)))
le,re = None,None
seq = list_seq_file[0]
list_eye_file = delpoint(root+'/'+seq,os.listdir(root+'/'+seq))
le_tmp,re_tmp =[],[]
for l in list_eye_file:
    if 'le' in l or 'left' in l.lower():
        le = l
    if 're' in l or 'right' in l.lower():
        re = l
assert  le and re ,'left eye and right eye image not exist'
le_tmp = prune_point(sorted(os.listdir(root+'/'+seq+'/'+le+'/'+image_dir)))
re_tmp = prune_point(sorted(os.listdir(root+'/'+seq+'/'+re+'/'+image_dir)))
le_res,re_res = [root+'/'+seq+'/'+le+'/'+image_dir+'/'+l for l in le_tmp],[root+'/'+seq+'/'+re+'/'+image_dir+'/'+l for l in re_tmp]
# wenli1 =cv2.imread('/home/rg0775/QingHong/dataset/gma_datasets/ghchen/wenli/wenli1.jpg')
# wenli1 = np.transpose(wenli1,(1,0,2))
# wenli2 =cv2.imread('/home/rg0775/QingHong/dataset/gma_datasets/ghchen/wenli/wenli2.jpg')
# blank = np.zeros((2000,2000,3))
mori =cv2.imread('/home/rg0775/QingHong/dataset/gma_datasets/ghchen/wenli/mori_b.jpg')
h_,w_,_ = mori.shape

def move(i,h,w,h_,w_,step_h,step_w):
    dis_h = abs(h_ - h)
    dis_w = abs(w_ - w)

    times_h = (dis_h-1) // step_h
    times_w = (dis_w-1) // step_w

    i_h = i % times_h*2
    i_w = i % times_w*2

    res_h = i_h * step_h if i_h <= times_h else (times_h*2 - i_h) * step_h
    res_w = i_w * step_w if i_w <= times_w else (times_w*2 - i_w) * step_w

    return res_h,res_w

def work(datas,wenli,name):
    for data in datas:
        image = cv2.imread(data)
        mask = cv2.imread(data.replace('/src/','/mask/'))[...,0]
        h,w,_ = image.shape
        wenli_ = wenli.copy()[:h,:w,:]
        pos = np.where(mask>0)
        wenli_[pos] = image[pos]
        targetpos = data.replace('/src/','/src_'+name+'/')
        mkdir(targetpos[:-targetpos[::-1].find('/')])
        cv2.imwrite(targetpos,wenli_)
    # n = len(datas)
    # for i in range(n):
    #     data = datas[i]
    #     image = cv2.imread(data)
    #     mask = cv2.imread(data.replace('/src/','/mask/'))[...,0]
    #     h,w,_ = image.shape
    #     wenli_ = wenli.copy()
    #     res_h,res_w = move(i,h,w,h_,w_,10,20)
    #     wenli_ = wenli_[res_h:res_h+h,res_w:res_w+w,:]
    #     pos = np.where(mask>0)
    #     wenli_[pos] = image[pos]
    #     targetpos = data.replace('/src/','/src_'+name+'/')
    #     mkdir(targetpos[:-targetpos[::-1].find('/')])
    #     cv2.imwrite(targetpos,wenli_)

for datas in [le_res,re_res]:
    # work(datas,wenli1,'wenli1')
    # work(datas,wenli2,'wenli2')
    work(datas,mori,'mori_b')