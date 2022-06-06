'''
Author: Qing Hong
Date: 2022-03-07 10:50:59
LastEditors: QingHong
LastEditTime: 2022-06-06 15:23:20
Description: file content
'''

import cv2
import numpy as np
import os,sys
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/core'
sys.path.insert(0, dir_mytest)
from tqdm import tqdm
from collections import defaultdict
import imageio
import torch
from myutil import *
from utils.utils import InputPadder
from utils import flow_viz
from network import RAFTGMA
from torch import distributed as dist
import datetime
# sudo apt-get install libopenexr-dev
# import OpenEXR
#import Imath
import re
FLOAT = None
# FLOAT = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))


'''
description: 对输入文件进行预处理,生成包含文件目录的dict文件
param {*} args
image_dir 文件名
return {*} 生成包含文件目录的dict文件
'''
def pre_treatment(args,image_dir):
    root = args.root
    n_limit = args.n_limit
    distributed_task = args.distributed_task
    assert os.path.isdir(root), '%s is not a valid directory' % root
    list_seq_file = delpoint(root,sorted(os.listdir(root)))
    final_res = defaultdict(list)
    for seq in list_seq_file:
        list_image = prune_point(sorted(os.listdir(root + '/' + seq + '/'+ image_dir)))
        if n_limit>0:
            tmp =  [root + '/' + seq + '/'+ image_dir +'/' + i for i in list_image[:n_limit]]
        else:
            tmp =  [root + '/' + seq + '/'+ image_dir +'/' + i for i in list_image]
        cur_rank_start = distributed_task[0]*len(tmp)//distributed_task[1]
        next_rank_start = (1+distributed_task[0])*len(tmp)//distributed_task[1]+1
        final_res[seq] =tmp[cur_rank_start:next_rank_start]
    return final_res

'''
description: 对深度计算的输入文件进行预处理
param {*} args
image_dir 左眼文件名
right_eye_file  右眼文件名
return {*} 生成包含文件目录的dict文件
'''
def pre_treatment_caldepth(args,image_dir,right_eye_file):
    root = args.root
    n_limit = args.n_limit
    distributed_task = args.distributed_task

    assert os.path.isdir(root), '%s is not a valid directory' % root
    list_seq_file = delpoint(root,sorted(os.listdir(root)))
    le,re = None,None
    le_res,re_res = defaultdict(list),defaultdict(list)
    for seq in list_seq_file:
        list_eye_file = delpoint(root+'/'+seq,os.listdir(root+'/'+seq))
        le_tmp,re_tmp =[],[]
        for l in list_eye_file:
            if 'le' in l or 'left' in l.lower():
                le = l
            if 're' in l or 'right' in l.lower():
                re = l
        assert  le and re ,'left eye and right eye image not exist'
        le_tmp = prune_point(sorted(os.listdir(root+'/'+seq+'/'+le+'/'+image_dir)))
        re_tmp = prune_point(sorted(os.listdir(root+'/'+seq+'/'+re+'/'+right_eye_file)))
        if n_limit>0:
            tmp1,tmp2 = [root+'/'+seq+'/'+le+'/'+image_dir+'/'+l for l in le_tmp[:n_limit]],[root+'/'+seq+'/'+re+'/'+right_eye_file+'/'+l for l in re_tmp[:n_limit]]
        else:
            tmp1,tmp2 = [root+'/'+seq+'/'+le+'/'+image_dir+'/'+l for l in le_tmp],[root+'/'+seq+'/'+re+'/'+right_eye_file+'/'+l for l in re_tmp]

        cur_rank_start = distributed_task[0]*len(tmp1)//distributed_task[1]
        next_rank_start = (1+distributed_task[0])*len(tmp1)//distributed_task[1]+1
        le_res[seq],re_res[seq] = tmp1[cur_rank_start:next_rank_start],tmp2[cur_rank_start:next_rank_start]
        
    return le_res,re_res


def load_exr(root,gt):
    list_exr_file = prune_point(os.listdir(root))
    final_res = defaultdict(list)
    for seq in list_exr_file:
        res = []
        exr_path = os.path.join(root,seq,gt)
        list_exr = sorted(prune_point(os.listdir(exr_path)))
        for i in range(len(list_exr)):
            image = imageio.imread(os.path.join(exr_path,list_exr[i]))
            res.append(image)
        final_res[seq] = res
    return final_res


def custom_refine(flow,zero_to_one=True):
    if flow.shape[2] == 4:
        return flow
    height,width = flow[...,0].shape
    #average value
    if zero_to_one:
        flow[...,0]/=width
        flow[...,1]/=-height
    else:
        flow[...,0]/=-width
        flow[...,1]/=height
    return flow
    

    #front masked area set to 0.5 and back to 1 

def save_file(save_path,flow,front_mask_,back_mask_,refine=False,savetype='exr',two_mask=False,using_mask='',zero_to_one=True):
        flow = flow.astype("float32")
        #append z axis
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        #refine
        if refine:
            flow = custom_refine(flow,zero_to_one=zero_to_one)
        #alpha 
        if front_mask_ and back_mask_:
            height,width,_ = flow.shape
            front_mask= imageio.imread(front_mask_)[...,0]
            back_mask = 255*257 - front_mask if not two_mask else imageio.imread(back_mask_)[...,0]
            if front_mask.max()>1:
                gt_front_mask = np.round(front_mask/(255*257))
            else:
                gt_front_mask = front_mask

            if back_mask.max()>1:
                gt_back_mask = np.round(back_mask/(255*257))
            else:
                gt_back_mask = back_mask

            gt_front_mask = gt_front_mask.astype(bool)
            gt_back_mask = gt_back_mask.astype(bool)
            alpha = np.ones((height,width)) * gt_front_mask *0.5 + np.ones((height,width)) * gt_back_mask
            alpha = alpha.reshape((height,width,1))
            flow = np.concatenate((flow,alpha),axis=2)

        #append
        if flow.shape[2] == 3 and savetype=='exr':
            flow = np.insert(flow,2,0,axis=2)
        
        # flow = depthToint16(flow)
        # saveUint16(save_path+'/' +key+ '_000'+str(i)+'.exr',flow)

        # flow = value[i]
        # if flow.shape[2] == 2:
        #     flow = np.insert(flow,2,0,axis=2)
        if savetype == 'exr':            imageio.imwrite(save_path,flow.astype("float32"))
        ##mv's depth need matting algorithm
        # if savetype == 'exr':            
            # imwrite(save_path,flow[...,0],flow[...,1],flow[...,2],alpha,None)
        else:
            imageio.imwrite(save_path,flow[...,:3])

def save_depth_file(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    # if zero_to_one:
    #     flow = -flow
    # flow = np.abs(flow)
    # flow = (flow + dp_value) / dp_value*2
    
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    flow = (flow + depth_range/2).clip(0, depth_range) / depth_range
    h,w = flow.shape
    res = np.ones((h,w,4))
    if reverse:
        flow *= -1
    res[...,3] = flow
    imageio.imwrite(save_path,res.astype("float32"))

'''
description: 使用mask对光流进行约束,去除mask之外的像素
param {*} flow 光流
param {*} mask_ mask文件
param {*} threshold 过滤阈值
param {*} mv_ref 颠倒过滤
return {*} 约束后的光流
'''
def restrain(flow,mask_,threshold,mv_ref):
    mask = imageio.imread(mask_)[...,0]
    if mv_ref:
        mask[np.where(mask<threshold)] = 0
        mask[np.where(mask>=threshold)] = 1
    else:
        mask[np.where(mask>threshold)] = 0
        mask[np.where(mask<=threshold)] = 1
    flow[np.where(mask<1)] =0
    return flow

'''
description: 光流计算代码
param {*} args
param {*} images 输入图片dict
param {*} append 保存名
param {*} front_mask_dict 前景mask的dict
param {*} back_mask_dict 背景mask的dict
param {*} zero_one 是否为01相位
param {*} using_mask 使用哪种mask
return {*} 光流结果地址dict
'''
def optical_flow(args,images,append,front_mask_dict,back_mask_dict,zero_one,using_mask):
    output = args.output
    threshold = args.threshold
    two_mask= args.char and args.bg
    mv_ref= args.mv_ref
    refine = args.refine
    savetype= args.savetype
    algorithm= args.algorithm
    DEVICE = args.DEVICE
    pass_mv= args.pass_mv
    use_tqdm= args.use_tqdm
    assert algorithm in ['gma','farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof','gma_patch'] or 'gma' in algorithm, 'not supported algorithm: %s' %algorithm
    res = defaultdict(list)
    model = None if 'gma' not in algorithm else get_model(DEVICE=DEVICE,args=args)
    for seq,seq_images in images.items():
        tmp = []
        total_range = range(len(seq_images)) if not use_tqdm else tqdm(range(len(seq_images)), desc='current sequence:{}'.format(seq))
        cost_time = 0
        for i in total_range:
            if using_mask == 'front':
                cur = mask_read(seq_images[i],front_mask_dict[seq][i],threshold=threshold,reverse=False,mv_ref=mv_ref)
                if i>0:
                    mask = front_mask_dict[seq][i-1]
            elif using_mask == 'bg':
                if two_mask:
                    cur = mask_read(seq_images[i],back_mask_dict[seq][i],threshold=threshold,reverse=False,mv_ref=mv_ref)
                    mask = back_mask_dict[seq][i]
                else:
                    cur = mask_read(seq_images[i],front_mask_dict[seq][i],threshold=threshold,reverse=True,mv_ref=mv_ref)
                    mask = front_mask_dict[seq][i]
                   ##bug here
            else:
                cur = cv2.imread(seq_images[i]) if not pass_mv else None 
                mask = None
            if  not os.path.exists(output+'/'+seq+'/'+append):
                os.makedirs(output+'/'+seq+'/'+append)
            name =getname(seq_images[i])
            if i == 0 :
                pre = cur
                continue
            fm = None
            bm = None
            if not using_mask=='None':
                fm = front_mask_dict[seq][i-1] if zero_one else front_mask_dict[seq][i]
                if two_mask:
                    bm = back_mask_dict[seq][i-1] if zero_one else back_mask_dict[seq][i]
            if zero_one:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1])-1,8)+ '.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i-1)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    start_time = datetime.datetime.now()
                    opt = optical_flow_algo(pre,cur,algorithm,DEVICE,model)
                    cost_time += (datetime.datetime.now()-start_time).microseconds
                    if args.restrain:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=True)
            else:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1]),8)+'.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    start_time = datetime.datetime.now()
                    opt = optical_flow_algo(cur,pre,algorithm,DEVICE,model)
                    cost_time += (datetime.datetime.now()-start_time).microseconds
                    if args.restrain:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=False)
            pre = cur
        if args.time_cost and args.use_tqdm:
            print('cost times:{:.1f}s'.format(cost_time/100000))
        res[seq] = tmp
    return res


def optical_flow_mask(images,output,append,front_mask_dict,back_mask_dict,threshold=40,two_mask = False,mv_ref=False,refine=False,savetype='exr',algorithm  = 'farneback',zero_one = True,using_mask = 'front',DEVICE='cpu',args=None,pass_mv=False,use_tqdm=False):
    assert algorithm in ['gma','farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof','gma_patch'] or 'gma' in algorithm, 'not supported algorithm: %s' %algorithm
    res = defaultdict(list)
    def getname(image):
        tmp = image.split('/')[-1]
        tmp = tmp[:-1-tmp[::-1].find('.')]
        tmp = tmp[-tmp[::-1].find('.'):]
        return tmp
    model = None if 'gma' not in algorithm else get_model(DEVICE=DEVICE,args=args)
    for seq,seq_images in images.items():
        tmp = []
        total_range = range(len(seq_images)) if not use_tqdm else tqdm(range(len(seq_images)), desc='current sequence:{}'.format(seq))
        for i in total_range:
            cur = mask_read(seq_images[i],front_mask_dict[seq][i],threshold=threshold,reverse=False,mv_ref=mv_ref)
            cur_full = cv2.imread(seq_images[i])
            if i>0:
                mask = front_mask_dict[seq][i-1]
            if  not os.path.exists(output+'/'+seq+'/'+append):
                os.makedirs(output+'/'+seq+'/'+append)
            name =getname(seq_images[i])
            if i == 0 :
                pre = cur
                pre_full = cur_full
                continue
            fm = None
            bm = None
            # if not using_mask=='None':
            #     fm = front_mask_dict[seq][i-1] if zero_one else front_mask_dict[seq][i]
            #     if two_mask:
            #         bm = back_mask_dict[seq][i-1] if zero_one else back_mask_dict[seq][i]
            if zero_one:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1])-1,8)+ '.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i-1)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    opt = optical_flow_algo(pre,cur_full,algorithm,DEVICE,model)
                    if mask:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=True)
            else:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1]),8)+'.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    opt = optical_flow_algo(cur,pre_full,algorithm,DEVICE,model)
                    if mask:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=False)
            pre = cur
            pre_full = cur_full
        res[seq] = tmp
    return res

def optical_flow_qcom(images,output,append,front_mask_dict,back_mask_dict,threshold=40,two_mask = False,mv_ref=False,refine=False,savetype='exr',algorithm  = 'farneback',zero_one = True,using_mask = 'front',DEVICE='cpu',args=None,pass_mv=False,use_tqdm=False):
    assert algorithm in ['gma','farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof','gma_patch'] or 'gma' in algorithm, 'not supported algorithm: %s' %algorithm
    res = defaultdict(list)
    model = None if 'gma' not in algorithm else get_model(DEVICE=DEVICE,args=args)
    for seq,seq_images in images.items():
        tmp = []
        total_range = range(len(seq_images)) if not use_tqdm else tqdm(range(len(seq_images)), desc='current sequence:{}'.format(seq))
        for i in total_range:
            cur = mask_read(seq_images[i],front_mask_dict[seq][i],threshold=threshold,reverse=False,mv_ref=mv_ref)
            cur_full = cv2.imread(seq_images[i])
            if i>0:
                mask = front_mask_dict[seq][i-1]
            if  not os.path.exists(output+'/'+seq+'/'+append):
                os.makedirs(output+'/'+seq+'/'+append)
            name =getname(seq_images[i])
            if i == 0 :
                pre = cur
                pre_full = cur_full
                continue
            fm = None
            bm = None
            if not using_mask=='None':
                fm = front_mask_dict[seq][i-1] if zero_one else front_mask_dict[seq][i]
                if two_mask:
                    bm = back_mask_dict[seq][i-1] if zero_one else back_mask_dict[seq][i]
            if zero_one:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1])-1,8)+ '.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i-1)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    opt = optical_flow_algo(pre,cur_full,algorithm,DEVICE,model)
                    if mask:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=True)
            else:
                tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(int(re.findall(r'\d+', name)[-1]),8)+'.' +savetype
                # tmp_file = output+'/'+seq+'/'+append+'/mv_'+appendzero(i)+ '.' +savetype
                tmp.append(tmp_file)
                if not pass_mv:
                    opt = optical_flow_algo(cur,pre_full,algorithm,DEVICE,model)
                    if mask:opt = restrain(opt,mask,threshold,mv_ref)
                    save_file(tmp_file,opt,fm,bm,refine=refine,savetype=savetype,two_mask=two_mask,using_mask=using_mask,zero_to_one=False)
            pre = cur
            pre_full = cur_full
        res[seq] = tmp
    return res

def mask_read(image,mask_file,threshold=40,reverse=False,mv_ref=False):
    res = cv2.imread(image)
    mask = imageio.imread(mask_file)[...,0]
    if reverse:
        mask = 65535 - mask
    if mv_ref:
        res[np.where(mask<threshold)] = 0
    else:
        res[np.where(mask>threshold)] = 0
    return res

def optical_flow_algo(pre,cur,algo='farneback',DEVICE='cpu',model=None):
    need_gray = False
    if algo in ['farneback','deepflow','disflow']:
        need_gray = True
    if need_gray:
        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    
    if algo =='farneback':
        flow = cv2.calcOpticalFlowFarneback(prev=pre, next=cur, flow=None, pyr_scale=0.5, levels=5,
                                            winsize=15,
                                            iterations=3, poly_n=3, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    elif algo == 'deepflow':
        inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(pre, cur, None)

    elif algo == 'simpleflow':
        flow = cv2.optflow.calcOpticalFlowSF(pre, cur, 2, 2, 4)

    elif algo == 'sparse_to_dense_flow':
        flow = cv2.optflow.calcOpticalFlowSparseToDense(pre, cur)

    elif algo == 'pca_flow':
        inst = cv2.optflow.createOptFlow_PCAFlow()
        flow = inst.calc(pre, cur, None)

    elif algo == 'rlof':
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(pre, cur,None)
    #388 584  392 584
    elif algo =='gma':
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            flow_low, flow_up = model(pre, cur, iters=12, test_mode=True)
        # flow = gma_viz(flow_low, flow_up)
        flow = flow_up.squeeze(0).cpu().detach().numpy()
        flow = np.transpose(padder.unpad(flow),(1,2,0))

    elif algo =='gma_resize':
        pre = cv2.resize(pre,(pre.shape[1]//2,pre.shape[0]//2))
        cur = cv2.resize(cur,(cur.shape[1]//2,cur.shape[0]//2))
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            flow_low, flow_up = model(pre, cur, iters=12, test_mode=True)
        # flow = gma_viz(flow_low, flow_up)
        flow = flow_up.squeeze(0).cpu().detach().numpy()
        flow = np.transpose(padder.unpad(flow),(1,2,0))

    elif algo =='gma_resize_quad':
        pre = cv2.resize(pre,(pre.shape[1]//4,pre.shape[0]//4))
        cur = cv2.resize(cur,(cur.shape[1]//4,cur.shape[0]//4))
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            flow_low, flow_up = model(pre, cur, iters=12, test_mode=True)
        # flow = gma_viz(flow_low, flow_up)
        flow = flow_up.squeeze(0).cpu().detach().numpy()
        flow = np.transpose(padder.unpad(flow),(1,2,0))
    
    elif algo =='gma_resize_oct':
        pre = cv2.resize(pre,(pre.shape[1]//8,pre.shape[0]//8))
        cur = cv2.resize(cur,(cur.shape[1]//8,cur.shape[0]//8))
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            flow_low, flow_up = model(pre, cur, iters=12, test_mode=True)
        # flow = gma_viz(flow_low, flow_up)
        flow = flow_up.squeeze(0).cpu().detach().numpy()
        flow = np.transpose(padder.unpad(flow),(1,2,0))

    elif algo =='gma_4k':
        #patch number, if  partition ==2, then image will devide into 4 image
        partition = 2
        #cover size
        cover_size_h = 8 + (8 - (pre.shape[0]//2)%8)
        cover_size_w = 8 + (8 - (pre.shape[1]//2)%8)

        def regroup(data):
            h,w,_ = data.shape
            assert min(h,w)// partition > cover_size_w and min(h,w)// partition > cover_size_h, 'cover size can not larger than image size!'

            res = np.zeros((4,h//2+cover_size_h,w//2+cover_size_w,3))
            # for i in range(partition**2):
            #     res[i,...] = data[max(0,i*hh):(i+1)*hh]
            #先默认partition=2 写死，后续要扩展再拓展代码
            res[0] = data[:h//2+cover_size_h,:w//2+cover_size_w,:]
            res[1] = data[:h//2+cover_size_h,w//2-cover_size_w:,:]
            res[2] = data[h//2-cover_size_h:,:w//2+cover_size_w:,:]
            res[3] = data[h//2-cover_size_h:,w//2-cover_size_w:,:]
            # res = np.zeros((partition**2,hh,ww,3))
            # res[0] = data[:hh,:ww,:]
            # res[1] = data[hh:,ww:,:]
            # res[2] = data[:hh,:ww,:]
            # res[3] = data[hh:,ww:,:]
            res = torch.from_numpy(res).permute(0, 3, 1, 2).float()
            return res

        def restore(data):
            #4,2,594 1056
            b,c,hhh,www = data.shape
            h = (hhh - cover_size_h) * 2
            w = (www - cover_size_w) * 2
            res = np.zeros((2,h,w))
            dup = np.zeros((h,w))

            res[:,:h//2+cover_size_h,:w//2+cover_size_w] = data[0]
            res[:,:h//2+cover_size_h,w//2-cover_size_w:] = data[1]
            res[:,h//2-cover_size_h:,:w//2+cover_size_w:] = data[2]
            res[:,h//2-cover_size_h:,w//2-cover_size_w:] = data[3]

            ##加权 求平均
            dup[:h//2+cover_size_h,:w//2+cover_size_w]+= 1
            dup[:h//2+cover_size_h,w//2-cover_size_w:] += 1
            dup[h//2-cover_size_h:,:w//2+cover_size_w:]  += 1
            dup[h//2-cover_size_h:,w//2-cover_size_w:] += 1

            res[0,...] = res[0,...] / dup
            res[1,...] = res[1,...] / dup
            return  np.transpose(res,(1,2,0))

        pre = regroup(pre).to(DEVICE)
        cur = regroup(cur).to(DEVICE)
        flow = []
        with torch.no_grad():
            for i in range(4):
                flow_low, flow_up = model(pre[i].unsqueeze(0), cur[i].unsqueeze(0), iters=12, test_mode=True)
                flow_ = flow_up.squeeze(0).cpu().detach().numpy().astype('float32')
                flow.append(flow_)
        # flow = gma_viz(flow_low, flow_up)
        flow = np.stack(flow)
        flow = restore(flow)

    return flow


# def get_model(DEVICE,args):
#     model = torch.nn.DataParallel(RAFTGMA(args))
#     model.load_state_dict(torch.load(args.model,map_location=DEVICE))
#     print(f"Loaded checkpoint at {args.model}")
#     model = model.module
#     model.to(DEVICE)
#     model.eval()
#     return model


#524新加 并行
def get_model(DEVICE,args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model,map_location=DEVICE))
    #model.load_state_dict(torch.load(args.model,map_location=DEVICE).state_dict())
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    # local_rank = torch.distributed.get_rank()
    # model = model.cuda(local_rank)
    # torch.cuda.set_device(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                 device_ids=[0, 1],
    #                                                 output_device=local_rank,
    #                                                 find_unused_parameters=False,
    #                                                 broadcast_buffers=False)
    return model


def gma_viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().detach().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().detach().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    return flo

def data_refine(data):
    if data.shape[2] == 2:
        data = np.insert(data,2,0,axis=2)
        data = np.insert(data,3,0,axis=2)
    elif data.shape[2]==3:
        data = np.insert(data,3,0,axis=2)
    return data

def appendzero(a,length=6):
    res = str(a)
    while(len(res)<length):
        res = '0'+ res
    return res

def getname(image):
        tmp = image.split('/')[-1]
        tmp = tmp[:-1-tmp[::-1].find('.')]
        tmp = tmp[-tmp[::-1].find('.'):]
        return tmp

'''
description: 光流(计算深度)
param {*} args
param {*} left 左眼图像dict
param {*} right 右眼图像dict
param {*} mask_ mask(若有) dict
param {*} append 保存名
param {*} zero_to_one 对应01相位
param {*} reverse 是否翻转深度(像素值取反)
return {*} 深度地址dict
'''
def optical_flow_depth(args,left,right,mask_,append,zero_to_one,reverse=False):
    algorithm = args.algorithm
    output = args.output
    mv_ref = args.mv_ref
    depth_range = args.depth_range
    threshold = args.threshold
    DEVICE = args.DEVICE
    export_half = args.export_half
    use_tqdm = args.use_tqdm

    res,res_ = defaultdict(list),defaultdict(list)
    model = None if 'gma' not in algorithm  else get_model(DEVICE=DEVICE,args=args)
    for seq,seq_images in left.items():
        tmp,tmp_ = [],[]
        l,r = left[seq],right[seq]
        mkdir(output+'/'+seq+'/'+append)
        if export_half:
            mkdir(output+'/'+seq+'/'+append+'_half')
        total_range = range(len(l)) if not use_tqdm else tqdm(range(len(l)), desc='current sequence:{}'.format(seq))
        for i in total_range:
            image1,image2 = cv2.imread(l[i]),cv2.imread(r[i])
            if mask_:
                mask = imageio.imread(mask_[seq][i])[...,0]
                if mv_ref:
                    mask[np.where(mask<threshold)] = 0
                    mask[np.where(mask>=threshold)] = 1
                else:
                    mask[np.where(mask<=threshold)] = 1
                    mask[np.where(mask>threshold)] = 0
                image1[np.where(mask<1)] = 0
                image2[np.where(mask<1)] = 0
            flow = optical_flow_algo(image1,image2,algorithm,DEVICE,model)[...,0] #x axis
            if mask_:
                flow *= mask
            name =getname(seq_images[i])
            tmp_file = output+'/'+seq+'/'+append+'/depth_'+appendzero(int(re.findall(r'\d+', name)[-1])-1,8)+ '.exr' 
            tmp.append(tmp_file)
            save_depth_file(tmp_file,flow,zero_to_one,half=False,depth_range=depth_range,reverse=reverse)
            if export_half:
                tmp_file_ = output+'/'+seq+'/'+append+'_half/depth_'+appendzero(i,8)+ '.exr' 
                tmp_.append(tmp_file_)
                save_depth_file(tmp_file_,flow,zero_to_one,half=True,depth_range=depth_range,reverse=reverse)
                res_[seq] = tmp_
        res[seq] = tmp
    return res,res_


'''
description: 融合光流和深度, rgba 对应 x,y,z的mv以及depth
param {*} args
param {*} flow 光流结果dict
param {*} depth 深度结果dict
param {*} append 保存名
param {*} zero_to_one 是否01相位
param {*} limited_result 是否对结果进行约束
return {*}
'''
def merge_depth(args,flow,depth,append,zero_to_one=True,limited_result=None):
    output = args.output
    use_tqdm = args.use_tqdm
    fusion_mode = args.fusion_mode
    assert fusion_mode.lower() in ['none','normal','inter'] , 'fusion mode error,ony None,Normal,Inter can be used'
    #depth =帧号
    res = defaultdict(list)
    for seq in flow.keys():
        tmp = []
        f = flow[seq]
        mkdir(output+'/'+seq+'/'+append)
        dd = None
        total_range = range(len(f)) if not use_tqdm else tqdm(range(len(f)), desc='current sequence:{}'.format(seq))
        for i in total_range:
            ff = imageio.imread(f[i])
            # ff = imread(f[i])
            if fusion_mode.lower()=='inter':
                #depth =帧号
                # if i==0:
                #     dd = imageio.imread(depth[seq][0])[...,3]
                # else:
                #     dd = imageio.imread(depth[seq][i-1])[...,3]
                if zero_to_one:
                    dd = imageio.imread(depth[seq][i])[...,3]
                else:
                    dd = imageio.imread(depth[seq][i+1])[...,3]
                ff[...,3] = dd
            elif fusion_mode.lower()=='normal':#保存mask的融合在前面做过了 这里就不做处理 取和alpha一样的即可
                # dd = ff[...,3]
                pass
            else:
                ff[...,3] = 0
                # pass
            if limited_result:
                mask = cv2.imread(limited_result[seq][i])[...,0]
                ff[np.where(mask==0)] = 0
            save_path = output+'/'+seq+'/'+append+'/mvd_'+re.findall(r'\d+', (f[i].split('/')[-1]))[-1]+ '.exr'
            imageio.imwrite(save_path,ff.astype("float32"))
            # imageio.imwrite(save_path,ff[...,0],ff[...,1],ff[...,2],ff[...,3],dd)
        res[seq] = tmp
    return res

def imwrite(save_path,image):
    r,g,b,a,d = [image[...,i] for i in range(5)]
    imwrite(save_path,r,g,b,a,d)

def imwrite(save_path,r,g,b,a,d):
    h,w = r.shape
    if not a:
        a=np.zeros((h,w))
    if not d:
        d=np.zeros((h,w))
    hd = OpenEXR.Header(h,w)
    hd['channels'] = {'B': FLOAT, 'G': FLOAT, 'R': FLOAT,'A': FLOAT,'D': FLOAT}
    exr = OpenEXR.OutputFile(save_path,hd)
    exr.writePixels({'R':r.tobytes(),'G':g.tobytes(),'B':b.tobytes(),'A':a.tobytes(),'D':d.tobytes()})
# a,b = pre_treatment('/Users/qhong/Documents/data/test_data','player','video')

def gma_demo():
    pass

