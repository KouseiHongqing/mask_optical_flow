'''
Author: Qing Hong
Date: 2022-03-07 10:50:59
LastEditors: QingHong
LastEditTime: 2022-03-10 19:23:04
Description: file content
'''

from collections import defaultdict
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import logging
import time
import configparser
import imageio
from EPE import *
from myutil import *
from algo import * 
import argparse

dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/core'
sys.path.insert(0, dir_mytest)

dir_3rd = os.path.dirname(os.path.abspath(__file__))+'/3rd/RobustVideoMatting'
sys.path.insert(0, dir_3rd)


##position
cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)
cur_path = sys.argv[0][:-sys.argv[0][::-1].find('/')]
if 'site-package' in cur_path:
    cur_path = ''
elif  cur_path.lower() in ['c','d']:
    cur_path = 'D:/python_repository/mask_optical_flow/'
elif cur_path.lower()=='m':
    cur_path = ''
##log
if  not os.path.exists(cur_path+'../logs'):
        os.makedirs(cur_path+'../logs')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s', #时间 文件名 line:行号  levelname logn内容
                    datefmt='%d %b %Y,%a %H:%M:%S', #日 月 年 ，星期 时 分 秒
                    filename= cur_path+'../logs/mylog{}.log'.format(cur_time),
                    filemode='w')

config_file = sys.argv[1]
# config_file = 'config'
sys.argv.pop(0)

def init(output):
    print('result file initiated')
    ##前景
    if os.path.exists(output):
        shutil.rmtree(output)

def fusion_mv(front_dict,front_mask_dict,back_dict,back_mask_dict,threshold=30,mv_ref=False):
    res = defaultdict(list)
    for seq in front_dict.keys():
        front,front_mask,back,back_mask = front_dict[seq],front_mask_dict[seq],back_dict[seq],back_mask_dict[seq]
        print('current sequence:{}'.format(seq))
        tmp = []
        for i in tqdm(range(len(front))):
            f,fm,b,bm = front[i],front_mask[i],back[i],back_mask[i]
            if mv_ref:
                f[np.where(fm<threshold)] = 0
                b[np.where(bm<threshold)] = 0
            else:
                f[np.where(fm>threshold)] = 0
                b[np.where(bm>threshold)] = 0
            fusion = f+b
            tmp.append(fusion)
        res[seq] = tmp
    return res

#传参用
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",default='checkpoints/gma-sintel.pth')
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    return args


def save_dict_file(filename,data):
    with open(filename,'w') as f:
            f.write('average EPE: {:.2f}, average accuracy :{:.2f}% \n\n'.format(data[0],data[1]*100))


def evaluation(source_datas,ground_truth_datas,masks_dict):
    cal_res = {}
    for key in source_datas.keys():
        print('evaluate:{}'.format(key))
        source_data = source_datas[key]
        target_data = ground_truth_datas[key]
        masks = masks_dict[key]
        n = len(source_data)
        w,h,_ = source_data[0].shape
        
        #first frame is passed
        epe = 0
        acc = 0
        start = 0 if n ==1 else 1
        for i in tqdm(range(start,n)):
            #set groud truth mask  predict mask here is same with ground truth mask
            mask = masks[i]
            if mask.max()>1:
                gt_mask = np.round(mask/255)
            else:
                gt_mask = mask
            gt_mask = gt_mask.astype(bool)
            pd_mask = gt_mask
            td = target_data[i]
            height,width = td[...,0].shape
            # refine average value
            td[...,0]*=width
            td[...,1]*=height

            epe_,acc_ = flow_kitti_mask_error(td[...,0],td[...,1],gt_mask,source_data[i][...,0],source_data[i][...,1],pd_mask)
            epe+=epe_
            acc+=acc_
        if n == 1:
            cal_res[key] = (epe,acc)
        else:
            cal_res[key] = (epe/(n-1),acc/(n-1))
    return cal_res


##载入config文件
config = configparser.ConfigParser()
config.read(cur_path + config_file, encoding="utf-8")

image_file = config.get('opticalflow','image_file')
front_mask_file = config.get('opticalflow','front_mask_file')
back_mask_file = config.get('opticalflow','back_mask_file')
rm_old_res = config.getboolean('opticalflow','rm_old_res')
algorithm = config.get('opticalflow','algorithm')
threshold = config.getint('opticalflow','threshold')
threshold_multiplier = config.getint('opticalflow','threshold_multiplier')
mv_ref = config.getboolean('opticalflow','mv_ref')
evaluate_epe = config.getboolean('opticalflow','evaluate_epe')
ground_truth_file = config.get('opticalflow','ground_truth_file')
two_mask = config.getboolean('opticalflow','two_mask')
refine = config.getboolean('opticalflow','refine')
root = config.get('opticalflow','root')
output = config.get('opticalflow','output')
cal_depth = config.getboolean('opticalflow','cal_depth')
right_eye_file = config.get('opticalflow','right_eye_file')
right_eye_mask_file = config.get('opticalflow','right_eye_mask_file')
savetype = config.get('opticalflow','savetype')
enable_merge_depth= config.getboolean('opticalflow','merge_depth')
dp_value = config.getint('opticalflow','dp_value')
char= config.getboolean('opticalflow','char')
bg= config.getboolean('opticalflow','bg')
no_mask = config.getboolean('opticalflow','no_mask')
gma_weight_file = config.get('opticalflow','gma_weight_file')
enable_dump = config.getboolean('opticalflow','enable_dump')
load_dump = config.getboolean('opticalflow','load_dump')
fusion_mode = config.get('opticalflow','fusion_mode')   
matting = config.getboolean('opticalflow','matting')
rm_ori_res = config.getboolean('opticalflow','rm_ori_res')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVICE='cpu'
args = None
if 'gma' in algorithm:
    import torch
    from network import RAFTGMA
    from utils import flow_viz
    from utils.utils import InputPadder
    args = get_args()
    args.mode = gma_weight_file
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold*=threshold_multiplier
if rm_old_res:
    # 文件夹出初始化
    init(output)

print(algorithm)

##预处理文件
print('current algorithm is : %s' % algorithm)
print('now loading image file and mv file')
if no_mask:
    front_mask = None
    bg_mask = None
    if cal_depth:
        image,right_eye_image= pre_treatment_caldepth(root,image_file,right_eye_file)
    else:
        image= pre_treatment(root,image_file)
else:
    if cal_depth: ##计算深度 通过左右眼方式
        print('calculate the depth')
        image,right_eye_image= pre_treatment_caldepth(root,image_file,right_eye_file)
        front_mask,right_eye_mask = pre_treatment_caldepth(root,front_mask_file,right_eye_file)
        bg_mask = {}
        if two_mask:
            bg_mask,right_eye_bg_mask = pre_treatment_caldepth(root,back_mask_file,right_eye_file)
    else:
        image= pre_treatment(root,image_file)
        front_mask = pre_treatment(root,front_mask_file)
        bg_mask = {}
        if two_mask:
            bg_mask = pre_treatment(root,back_mask_file)

print('pre-treatment finished,the number of sequence is {}, they are {}'.format(len(image.keys()),image.keys()))
print('now starting {} algorithm'.format(algorithm))
# for algorithm in ['farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof']:
if char and not no_mask:
    print('front mv0:')
    mv0_front_result = optical_flow(image,output,'{}_char_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='front')
    if cal_depth: mv0_front_result_right = optical_flow(right_eye_image,output,'{}_right_Char_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='front')

    print('front mv1:')
    mv1_front_result = optical_flow(image,output,'{}_char_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='front')
    if cal_depth:mv1_front_result_right = optical_flow(right_eye_image,output,'{}_right_Char_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='front')
if bg and not no_mask:
    print('back mv0:')
    mv0_back_result = optical_flow(image,output,'{}_bg_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='bg')
    if cal_depth:mv0_back_result = optical_flow(right_eye_image,output,'{}_right_bg_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='bg')

    print('back mv1:')
    mv1_back_result = optical_flow(image,output,'{}_bg_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='bg')
    if cal_depth:mv1_back_result_right = optical_flow(right_eye_image,output,'{}_right_bg_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='bg')

if no_mask:
    print('original mv0')
    mv0_origin_result = optical_flow(image,output,'{}_original_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='None',DEVICE=DEVICE,args=args)
    if cal_depth:mv0_origin_result_right = optical_flow(right_eye_image,output,'{}_right_original_mv0'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=True,using_mask='None',DEVICE=DEVICE,args=args)

    print('original mv1')
    mv1_origin_result = optical_flow(image,output,'{}_original_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='None',DEVICE=DEVICE,args=args)
    if cal_depth:mv1_origin_result_right = optical_flow(right_eye_image,output,'{}_right_original_mv1'.format(algorithm),front_mask,bg_mask,threshold=threshold,two_mask=two_mask,mv_ref=mv_ref,refine=refine,savetype=savetype,algorithm=algorithm,zero_one=False,using_mask='None',DEVICE=DEVICE,args=args)




if cal_depth: 
    print('depth:')
    if no_mask:
        depth_left_right = optical_flow_depth(image,right_eye_image,None,algorithm,output,'left_right_depth',zero_to_one=True,dp_value=dp_value,mv_ref=mv_ref,DEVICE=DEVICE,args=args)
        depth_right_left = optical_flow_depth(right_eye_image,image,None,algorithm,output,'right_left_depth',zero_to_one=True,dp_value=dp_value,mv_ref=mv_ref,DEVICE=DEVICE,args=args)
    else:    
        depth_left_right_char = optical_flow_depth(image,right_eye_image,front_mask,algorithm,output,'left_right_char_depth',zero_to_one=True,dp_value=dp_value,mv_ref=mv_ref)
        depth_right_left_char = optical_flow_depth(right_eye_image,image,front_mask,algorithm,output,'right_left_char_depth',zero_to_one=False,dp_value=dp_value,mv_ref=mv_ref)
        depth_left_right_bg = optical_flow_depth(image,right_eye_image,bg_mask,algorithm,output,'left_right_bg_depth',zero_to_one=True,dp_value=dp_value,mv_ref=mv_ref)
        depth_right_left_bg = optical_flow_depth(right_eye_image,image,bg_mask,algorithm,output,'right_left_bg_depth',zero_to_one=False,dp_value=dp_value,mv_ref=mv_ref)
    
if enable_merge_depth:
    #for test
    if enable_dump and cal_depth:
        import pickle
        dump_file = os.path.join(cur_path,'dump')
        if rm_old_res and os.path.exists(dump_file):
            shutil.rmtree(dump_file)
        if  not os.path.exists(dump_file):
            os.makedirs(dump_file)
        dump_data = {'mv0_origin_result':mv0_origin_result,'mv1_origin_result':mv1_origin_result,'depth_left_right':depth_left_right,'depth_right_left':depth_right_left}
        with open(dump_file+'/mydump','wb') as f:
            pickle.dump(dump_data,f)
    if load_dump:
        with open(dump_file+'/mydump','rb') as f:
            load_data = pickle.load(f)
        mv0_origin_result = load_data['mv0_origin_result']
        mv1_origin_result = load_data['mv1_origin_result']
        depth_left_right = load_data['depth_left_right']
        depth_right_left = load_data['depth_right_left']

    print('merge depth')
    if no_mask:
        merged_origin_mv0_lr = merge_depth(mv0_origin_result,depth_left_right,output,'{}_left_right_mergedepth_mv0'.format(algorithm),fusion_mode)
        merged_origin_mv1_lr = merge_depth(mv1_origin_result,depth_left_right,output,'{}_left_right_mergedepth_mv1'.format(algorithm),fusion_mode,zero_to_one=False)
        merged_origin_mv0_rl = merge_depth(mv0_origin_result_right,depth_right_left,output,'{}_right_left_mergedepth_mv0'.format(algorithm),fusion_mode)
        merged_origin_mv1_rl = merge_depth(mv1_origin_result_right,depth_right_left,output,'{}_right_left_mergedepth_mv1'.format(algorithm),fusion_mode,zero_to_one=False)
    else:
        merged_front_mv0 = merge_depth(mv0_front_result,depth_left_right,output,'{}_right_Char_mv0_depth'.format(algorithm),fusion_mode)
        merged_front_mv1 = merge_depth(mv0_front_result,depth_left_right,output,'{}_right_Char_mv1_depth'.format(algorithm),fusion_mode,zero_to_one=False)
        merged_bg_mv0 = merge_depth(mv0_front_result,depth_left_right,output,'{}_right_bg_mv0_depth'.format(algorithm),fusion_mode)
        merged_bg_mv1 = merge_depth(mv0_front_result,depth_left_right,output,'{}_right_Char_mv0_depth'.format(algorithm),fusion_mode,zero_to_one=False)
    
if rm_ori_res:
    for key in image:
        pat = os.path.join(output,key)
        for frag in  ['gma_original_mv0','gma_original_mv1','left_right_depth','right_left_depth','gma_right_original_mv0','gma_right_original_mv1']:
            shutil.rmtree(os.path.join(pat,frag),ignore_errors=True)

#     mv1_front_result = merge_depth(mv1_front_result,depth_mv_mask)
#     mv1_back_result = merge_depth(mv1_back_result,depth_mv_mask)
#     mv0_front_result = merge_depth(mv0_front_result,depth_mv_mask)
#     mv0_back_result = merge_depth(mv0_back_result,depth_mv_mask)
#     mv1_origin_result = merge_depth(mv1_origin_result,depth_mv)
#     mv0_origin__result = merge_depth(mv0_origin__result,depth_mv)

# ##融合
# mv1_fusion_result = fusion_mv(mv1_front_result,front_mask,mv1_back_result,back_mask,threshold,mv_ref)
# mv0_fusion_result = fusion_mv(mv0_front_result,front_mask,mv0_back_result,back_mask,threshold,mv_ref)

# #copy source file
# for key in image.keys():
#     for ls in os.listdir(os.path.join(root,key)):
#         if ls[0]!='.' and os.path.isdir(os.path.join(root,key,ls)):
#             shutil.copytree(os.path.join(root,key,ls),os.path.join(output,key,ls))
# print('result finished')

# #EPE
# if evaluate_epe:
#     print('start end point error evaluation')
#     ground_truth_datas = load_exr(root,ground_truth_file)
#     print('ground truth loaded')

#     ev_m1_front = evaluation(mv1_front_result,ground_truth_datas,front_mask)
#     ev_m1_back = evaluation(mv1_back_result,ground_truth_datas,front_mask)
#     ev_m1_fusion = evaluation(mv1_front_result,ground_truth_datas,front_mask)
#     ev_m1_origin = evaluation(mv1_origin_result,ground_truth_datas,front_mask)

#     for key in ev_m1_front.keys():
#         save_dict_file(os.path.join(output,key,'front.txt'),ev_m1_front[key])
#         save_dict_file(os.path.join(output,key,'back.txt'),ev_m1_back[key])
#         save_dict_file(os.path.join(output,key,'fusion.txt'),ev_m1_fusion[key])
#         save_dict_file(os.path.join(output,key,'origin.txt'),ev_m1_origin[key])

#     print('EPE finished')
# # print('')
# # print('Press any key to continue')
# os.system('pause')

