'''
Author: Qing Hong
Date: 2022-02-23 14:12:43
LastEditors: QingHong
LastEditTime: 2022-06-01 03:24:57
Description: Average End Point Error(average EPE)
'''
import numpy as np 
import cv2
from PIL import Image
import os,sys
from tqdm import tqdm
import imageio as iio
# iio.plugins.freeimage.download()


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w, h = np.fromfile(f, np.int32, count=2)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d 

def load_datas(source,mv='mv1',type = 'flo',limit=''):
    '''
    description: load all .flo file in target path  
    param {str} root: path to data file
    return {dict} : optical flow data sets
    '''    
    root = source +'/' +mv
    assert os.path.isdir(root), '%s is not a valid directory' % root
    res = {}
    list_root_ = os.listdir(root)
    list_root = []
    #prune the unvalid data
    for lr in list_root_:
        if lr[0] !='.':
            list_root.append(lr)

    for i in tqdm(range(len(list_root))):
        scene = list_root[i]
        tmp = []
        seqs = sorted(os.listdir(root + '/' +scene))
        n = len(seqs) ## for test
        for seq in seqs:
            if seq[0]=='.':
                continue
            if len(limit)>0 and n >1:
                if limit not in seq:
                    continue
            if type == 'flo':
                tmp.append(read_flo_file(root + '/' +scene+'/'+seq))
            elif type=='exr':
                tmp.append(np.array(iio.imread(root + '/' +scene+'/'+seq,'exr'))[...,:2])
            else:
                print(root + '/' +scene+'/'+seq)
                tmp.append(cv2.imread(root + '/' +scene+'/'+seq)[...,:2])
        res[list_root[i] ] = tmp
    return res    
    


def flow_kitti_mask_error(tu, tv, gt_mask, u, v, pd_mask):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param gt_mask: ground-truth mask

    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param pd_mask: estimated flow mask
    :return: End point error of the estimated flow
    """

    ##tau[0] genju tu lai kan
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    s_gt_mask = gt_mask[:]
    s_pd_mask = pd_mask[:]

    ind_valid = np.logical_and(s_gt_mask != 0, s_pd_mask != 0)
    n_total = np.sum(ind_valid)
    #替换过大的值
    def fix_data(data):
        data = np.where(data>=250,.0,data)
        data = np.where(data<=-250,.0,data)
        data = np.where(np.isnan(data),0,data)
        return data
    su = fix_data(su)
    sv = fix_data(sv)
    stu = fix_data(stu)
    stv = fix_data(stv)

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    mag = np.sqrt(stu**2 + stv**2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    mean_epe = np.mean(epe)
    mean_acc = 1 - (float(n_err) / float(n_total))
    return (mean_epe, mean_acc) 

def end_point_error(tu,tv,u,v):

    pass

def save_dict_file(filename,dic):
    with open(filename+'_res.txt','w') as f:
        for key,value in dic.items():
            f.write('{} average EPE: {:.2f}, average accuracy :{:.2f}% \n\n'.format(key,value[0],value[1]*100))


# source = '/Users/qhong/Documents/pythonhome/datas'
# target = '/Users/qhong/Documents/pythonhome/datas'

# # source = '//Users/qhong/Documents/data/2022'
# # target = '/Users/qhong/Documents/data/2022'
# algos  = ['farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof']
# # source_mv = 'mv2_farneback'
# # source_mv = 'mv2_rlof'
# target_mv = 'ground_truth'
# print('start load target mv data')
# target_datas = load_datas(target,target_mv)
# print('target data loaded')


# for source_mv in algos:
#     print('start load source mv data')
#     source_datas = load_datas(source,source_mv,type='flo',limit='03')
#     print('source data loaded')

#     assert len(source_datas) == len(target_datas), 'invalid datasets' 
#     cal_res = {}
#     print('start EPE evaluation')
#     for key in source_datas.keys():
#         print('evaluate:{}'.format(key))
#         source_data = source_datas[key]
#         target_data = target_datas[key]
#         n = len(source_data)
#         w,h,_ = source_data[0].shape
#         #set groud truth mask
#         gt_mask = np.ones((w,h))
#         pd_mask = np.ones((w,h))
#         #first frame is passed
#         epe = 0
#         acc = 0
#         start = 0 if n ==1 else 1
#         for i in tqdm(range(start,n)):
#             epe_,acc_ = flow_kitti_mask_error(target_data[i][...,0],target_data[i][...,1],gt_mask,source_data[i][...,0],source_data[i][...,1],pd_mask)
#             epe+=epe_
#             acc+=acc_
#         if n == 1:
#             cal_res[key] = (epe,acc)
#         else:
#             cal_res[key] = (epe/(n-1),acc/(n-1))
#     print('EPE finished')
#     save_dict_file('/Users/qhong/Documents/pythonhome/result/'+source_mv,cal_res)