import torch
import os,datetime
from torch import distributed as dist

# import os,sys
# import argparse
# dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/core'
# sys.path.insert(0, dir_mytest)
# from network import RAFTGMA
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'

# def init_distributed_mode():
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         gpu = int(os.environ['LOCAL_RANK'])
#     else:
#         print('Not using distributed mode')
#         distributed = False
#         return

#     distributed = True
#     dist_url = 'env://'
#     torch.cuda.set_device(gpu)
#     dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
#     print('| distributed init (rank {}): {}'.format(
#         rank, dist_url, flush=True))
#     dist.init_process_group(backend=dist_backend, init_method=dist_url,
#                             world_size=world_size, rank=rank)
#     dist.barrier()

# init_distributed_mode()

# # python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py
# # def get_args():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--model', help="restore checkpoint",default='checkpoints/gma-sintel.pth')
# #     parser.add_argument('--model_name', help="define model name", default="GMA")
# #     parser.add_argument('--path', help="dataset for evaluation")
# #     parser.add_argument('--num_heads', default=1, type=int,
# #                         help='number of heads in attention and aggregation')
# #     parser.add_argument('--position_only', default=False, action='store_true',
# #                         help='only use position-wise attention')
# #     parser.add_argument('--position_and_content', default=False, action='store_true',
# #                         help='use position and content-wise attention')
# #     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
# #     args = parser.parse_args()
# #     return args

# # args = get_args()

# # model = torch.nn.DataParallel(RAFTGMA(args))
# # model.load_state_dict(torch.load('checkpoints/gma-sintel.pth',map_location='cpu'))

# # torch.save(model, 'checkpoints/gma-sintel_N.pth', _use_new_zipfile_serialization=False)

# # git config --global user.email "kousei19920804@gmail.com"
# # git config --global user.name "KouseiHongqing"

# # apt-get install gnutls-bin
# # git config --global http.sslVerify false
# # git config --global http.postBuffer 1048576000
# ssh tao@10.7.13.107
# pxlwtc@226
# root = /tt/nas/AV1R3_V2_J2K
# # root = /Volumes/PXLW_INSSD_16TB/coral_tiffs_native/MMof_results/0311/sparse_to_dense_flow_3D/

# #the position of output
# output = /tt/nas/AV1R3_V2_J2K/R3_V2/MM
 
# ssh truecut@169.254.77.104
# TC@2020g

# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import imageio

# def appendzero(a,length=8):
#     res = str(a)
#     while(len(res)<length):
#         res = '0'+ res
#     return res
# TC@@2020g@
# def show_mv(n,a):
#     image = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/525_3d_output_05k/AVTR1_01_Jakewaking_3D/gma_resize_quad_left_right_mergedepth_mv1/mvd_'
#     image += appendzero(n)
#     image += '.exr'
#     img = imageio.imread(image)
#     Image.fromarray((img[...,3]*255*a).astype('uint8')).show()
    

# def show_depth(n,a):
#     image = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/525_3d_output_05k/AVTR1_01_Jakewaking_3D/gma_resize_quad_left_right_mergedepth_mv1/mvd_'
#     image += appendzero(n)
#     image += '.exr'
#     img = imageio.imread(image)
#     Image.fromarray((img[...,:3]*255*a).astype('uint8')).show()

# def show_image(n):
#     image = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/3d_image/AVTR1_01_Jakewaking_3D/le/image/AVTAR_01_Jakewaking_'
#     image += appendzero(n)
#     image += '.tiff'
#     img = cv2.imread(image)
#     Image.fromarray((img)).show()


# c = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/525_3d_output_05k/AVTR1_01_Jakewaking_3D/gma_resize_left_right_mergedepth_mv0/mvd_00000005.exr'

# d = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/525_3d_output_05k/AVTR1_01_Jakewaking_3D/left_right_depth/depth_00000005.exr'
# Image.fromarray((q[...,3]*255).astype('uint8'))
# aa(c)

import argparse
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

args = get_args()
args.nim = 'qwe'
print(args.nim)