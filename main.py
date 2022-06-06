'''
Author: Qing Hong
Date: 2022-03-07 10:50:59
LastEditors: QingHong
LastEditTime: 2022-06-06 14:48:49
Description: file content
'''
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import logging
import time,datetime
import configparser
import imageio
from EPE import *
from myutil import *
from algo import * 
import argparse
from torch import distributed as dist
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

def init_distributed_mode(cuda=False,backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])>1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if rank==0:
            print('using distributed mode, world_size is:{}'.format(world_size))
    else:
        print('Not using distributed mode')
        distributed = False
        return 0,1
    distributed = True
    dist_url = 'env://'
    if cuda:
        torch.cuda.set_device(local_rank)
    dist_backend = backend  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url, flush=True))
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()
    return local_rank,world_size

##load config
config = configparser.ConfigParser()
config.read(cur_path + config_file, encoding="utf-8")
args = get_args()
args.image_file = config.get('opticalflow','image_file')
args.front_mask_file = config.get('opticalflow','front_mask_file')
args.back_mask_file = config.get('opticalflow','back_mask_file')
args.algorithm = config.get('opticalflow','algorithm')
args.mt_backend = config.get('opticalflow','mt_backend')

#
args.mv_ref = config.getboolean('opticalflow','mv_ref')
args.evaluate_epe = config.getboolean('opticalflow','evaluate_epe')
args.ground_truth_file = config.get('opticalflow','ground_truth_file')
args.refine = config.getboolean('opticalflow','refine')
args.root = config.get('opticalflow','root')
args.output = config.get('opticalflow','output')
args.cal_depth = config.getboolean('opticalflow','cal_depth')
args.right_eye_file = config.get('opticalflow','right_eye_file')
args.right_eye_front_mask_file = config.get('opticalflow','right_eye_front_mask_file')
args.right_eye_back_mask_file = config.get('opticalflow','right_eye_back_mask_file')
#exr
args.savetype = config.get('opticalflow','savetype')
args.merge_depth= config.getboolean('opticalflow','merge_depth')
args.depth_range = config.getint('opticalflow','depth_range')
args.char= config.getboolean('opticalflow','char')
args.bg= config.getboolean('opticalflow','bg')
args.restrain = config.getboolean('opticalflow','restrain')
args.gma_weight_file = config.get('opticalflow','gma_weight_file')
args.enable_dump = config.getboolean('opticalflow','enable_dump')
args.load_dump = config.getboolean('opticalflow','load_dump')
args.fusion_mode = config.get('opticalflow','fusion_mode')   
args.matting = config.getboolean('opticalflow','matting')
args.rm_ori_res = config.getboolean('opticalflow','rm_ori_res')
args.export_half = config.getboolean('opticalflow','export_half')
args.n_limit = config.getint('opticalflow','n_limit')
args.gpu = config.get('opticalflow','gpu')
args.enable_limited = config.getboolean('opticalflow','enable_limited')
args.time_cost = config.getboolean('opticalflow','time_cost')
#refine threshold
args.threshold = config.getint('opticalflow','threshold') * config.getint('opticalflow','threshold_multiplier')
##跳过mv计算 测试代码不要修改 test parameter, donot edit
args.pass_mv = False
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = args.gpu.rstrip().split(',')
#os.environ['IMAGEIO_USERDIR'] = '/tt/nas/DSP_workdir/qhong/optical-flow/libx/imageio/freeimage'
DEVICE='cpu'
args.cur_rank = 1

##先决判断 启动bg_mask前要保证char_mask启用
if args.bg:
    assert args.char ,'make sure char_mask enabled'

if 'gma' in args.algorithm:
    import torch
    from network import RAFTGMA
    from utils import flow_viz
    from utils.utils import InputPadder
    args.mode = args.gma_weight_file
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.DEVICE = DEVICE
##mutl process
args.distributed_task = init_distributed_mode(torch.cuda.is_available(),args.mt_backend)
args.cur_rank = args.distributed_task[0] + 1
args.use_tqdm = True if args.cur_rank ==1 else False
##预处理文件
if args.cur_rank == 1:
    print('current algorithm is : %s' % args.algorithm)
    print('now loading image file and mv file')
if not (args.char or args.bg):
    front_mask = None
    bg_mask = None
    if args.cal_depth:
        image,right_eye_image= pre_treatment_caldepth(args,args.image_file,args.right_eye_file)
    else:
        image= pre_treatment(args,args.image_file)
else:
    if args.cal_depth: ##计算深度 通过左右眼方式
        print('calculate the depth')
        image,right_eye_image= pre_treatment_caldepth(args,args.image_file,args.right_eye_file)
        front_mask,right_eye_mask = pre_treatment_caldepth(args,args.front_mask_file,args.right_eye_file)
        bg_mask = {}
        if args.bg:
            bg_mask,right_eye_bg_mask = pre_treatment_caldepth(args,args.back_mask_file,args.right_eye_file)
    else:
        image= pre_treatment(args,args.image_file)
        front_mask = pre_treatment(args,args.front_mask_file)
        bg_mask = None
        if args.bg:
            bg_mask = pre_treatment(args,args.back_mask_file)

# #对光流结果进行约束,将结果限定在mask范围内
limited_result,limited_result_r = None,None
if args.enable_limited and args.cal_depth and args.char:
    limited_result,limited_result_r = pre_treatment_caldepth(args,args.front_mask_file,args.right_eye_file)
if args.cur_rank == 1:
    print('pre-treatment finished,the number of sequence is {}, they are {}'.format(len(image.keys()),image.keys()))
    print('now starting {} algorithm'.format(args.algorithm))
# for algorithm in ['farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof']:
# 计算前景mv结果
if args.char:
    if args.cur_rank == 1:print('front mv0:')
    mv0_front_result = optical_flow(args,image,'{}_char_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='front')
    if args.cal_depth: mv0_front_result_right = optical_flow(args,right_eye_image,'{}_right_Char_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='front')

    if args.cur_rank == 1:print('front mv1:')
    mv1_front_result = optical_flow(args,image,'{}_char_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='front')
    if args.cal_depth:mv1_front_result_right = optical_flow(args,right_eye_image,'{}_right_Char_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='front')
# 计算背景mv结果
if args.bg:
    if args.cur_rank == 1:print('back mv0:')
    mv0_back_result = optical_flow(args,image,'{}_bg_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='bg')
    if args.cal_depth:mv0_back_result = optical_flow(args,right_eye_image,'{}_right_bg_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='bg')

    if args.cur_rank == 1:print('back mv1:')
    mv1_back_result = optical_flow(args,image,'{}_bg_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='bg')
    if args.cal_depth:mv1_back_result_right = optical_flow(args,right_eye_image,'{}_right_bg_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='bg')
# 计算原始mv结果
if not (args.char or args.bg):
    if args.cur_rank == 1:
        print('original mv0')
    mv0_origin_result = optical_flow(args,image,'{}_left_original_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='None')
    if args.merge_depth:mv0_origin_result_right = optical_flow(args,right_eye_image,'{}_right_original_mv0'.format(args.algorithm),front_mask,bg_mask,zero_one=True,using_mask='None')
    if args.cur_rank == 1:
        print('original mv1')
    mv1_origin_result = optical_flow(args,image,'{}_left_original_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='None')
    if args.merge_depth:mv1_origin_result_right = optical_flow(args,right_eye_image,'{}_right_original_mv1'.format(args.algorithm),front_mask,bg_mask,zero_one=False,using_mask='None')
# 计算深度
if args.cal_depth: 
    if args.cur_rank == 1:
        print('depth:')
    if not (args.char or args.bg):
        depth_left_right,depth_left_right_half = optical_flow_depth(args,image,right_eye_image,None,'left_right_depth',zero_to_one=True)
        depth_right_left,depth_right_left_half = optical_flow_depth(args,right_eye_image,image,None,'right_left_depth',zero_to_one=False)
    else:    
        depth_left_right_char,_ = optical_flow_depth(args,image,right_eye_image,front_mask,'left_right_char_depth',zero_to_one=True)
        depth_right_left_char,_ = optical_flow_depth(args,right_eye_image,image,front_mask,'right_left_char_depth',zero_to_one=False)
        depth_left_right_bg,_ = optical_flow_depth(args,image,right_eye_image,bg_mask,'left_right_bg_depth',zero_to_one=True)
        depth_right_left_bg,_ = optical_flow_depth(args,right_eye_image,image,bg_mask,'right_left_bg_depth',zero_to_one=False)
# 混合深度并保存
if args.merge_depth:
    #for test
    if args.enable_dump and args.cal_depth:
        import pickle
        dump_file = os.path.join(cur_path,'dump')
        if  os.path.exists(dump_file):
            shutil.rmtree(dump_file)
        if  not os.path.exists(dump_file):
            os.makedirs(dump_file)
        dump_data = {'mv0_origin_result':mv0_origin_result,'mv1_origin_result':mv1_origin_result,'depth_left_right':depth_left_right,'depth_right_left':depth_right_left}
        with open(dump_file+'/mydump','wb') as f:
            pickle.dump(dump_data,f)
    if args.load_dump:
        with open(dump_file+'/mydump','rb') as f:
            load_data = pickle.load(f)
        mv0_origin_result = load_data['mv0_origin_result']
        mv1_origin_result = load_data['mv1_origin_result']
        depth_left_right = load_data['depth_left_right']
        depth_right_left = load_data['depth_right_left']
    if args.cur_rank == 1:
        print('merge depth')
    if not (args.char or args.bg):
        merged_origin_mv0_lr = merge_depth(args,mv0_origin_result,depth_left_right,'{}_left_right_mergedepth_mv0'.format(args.algorithm),limited_result=limited_result)
        merged_origin_mv1_lr = merge_depth(args,mv1_origin_result,depth_left_right,'{}_left_right_mergedepth_mv1'.format(args.algorithm),zero_to_one=False,limited_result=limited_result)
        merged_origin_mv0_rl = merge_depth(args,mv0_origin_result_right,depth_right_left,'{}_right_left_mergedepth_mv0'.format(args.algorithm),limited_result=limited_result_r)
        merged_origin_mv1_rl = merge_depth(args,mv1_origin_result_right,depth_right_left,'{}_right_left_mergedepth_mv1'.format(args.algorithm),zero_to_one=False,limited_result=limited_result_r)
        if args.export_half:
            merged_origin_mv0_lr_half = merge_depth(args,mv0_origin_result,depth_left_right_half,'{}_left_right_mergedepth_mv0_half'.format(args.algorithm),limited_result=limited_result)
            merged_origin_mv1_lr_half = merge_depth(args,mv1_origin_result,depth_left_right_half,'{}_left_right_mergedepth_mv1_half'.format(args.algorithm),zero_to_one=False,limited_result=limited_result)
            merged_origin_mv0_rl_half = merge_depth(args,mv0_origin_result_right,depth_right_left_half,'{}_right_left_mergedepth_mv0_half'.format(args.algorithm),limited_result=limited_result_r)
            merged_origin_mv1_rl_half = merge_depth(args,mv1_origin_result_right,depth_right_left_half,'{}_right_left_mergedepth_mv1_half'.format(args.algorithm),zero_to_one=False,limited_result=limited_result_r)
    else:
        merged_front_mv0 = merge_depth(args,mv0_front_result,depth_left_right,'{}_right_Char_mv0_depth'.format(args.algorithm),limited_result=limited_result)
        merged_front_mv1 = merge_depth(args,mv0_front_result,depth_left_right,'{}_right_Char_mv1_depth'.format(args.algorithm),zero_to_one=False,limited_result=limited_result)
        merged_bg_mv0 = merge_depth(args,mv0_front_result,depth_left_right,'{}_right_bg_mv0_depth'.format(args.algorithm),limited_result=limited_result_r)
        merged_bg_mv1 = merge_depth(args,mv0_front_result,depth_left_right,'{}_right_Char_mv0_depth'.format(args.algorithm),zero_to_one=False,limited_result=limited_result_r)
    
# if rm_ori_res and cur_rank == 1:
#     for key in image:
#         pat = os.path.join(output,key)
#         for ag in ['gma','gma_resize','gma_resize_oct','gma_resize_quad','gma_4k']:
#             for ag_att in ['left_original_mv0','left_original_mv1','right_original_mv0','right_original_mv1']:
#                 frag = ag+'_'+ag_att
#                 shutil.rmtree(os.path.join(pat,frag),ignore_errors=True)

print('======================================== worker {} is finished! ======================================= totalworker={}'.format(args.cur_rank,int(os.environ['WORLD_SIZE'])))
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

