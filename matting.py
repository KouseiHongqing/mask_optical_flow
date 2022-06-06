import os,sys
import torch
import cv2
import matplotlib.pyplot as plt
dir_3rd = os.path.dirname(os.path.abspath(__file__))+'/3rd/RobustVideoMatting'
sys.path.insert(0, dir_3rd)
from model import MattingNetwork


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE='cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MattingNetwork('resnet50').eval().to(DEVICE)
# model = torch.hub.load("PeterL1n/RobustVideoMatting","resnet50")
model.load_state_dict(torch.load('3rd/RobustVideoMatting/rvm_resnet50.pth'))

from inference import convert_video

sourcew = '/home/rg0775/QingHong/dataset/gma_datasets/mhyang/Titanic/clip01_left/src/'

convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source=sourcew,        # 视频文件，或图片序列文件夹
    output_type='png_sequence',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    #output_composition=sourcew+'../output',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_alpha=sourcew+'../mask',          # [可选项] 输出透明度预测
    #output_foreground="",     # [可选项] 输出前景预测
    output_video_mbps=4,             # 若导出视频，提供视频码率
    downsample_ratio=None,           # 下采样比，可根据具体视频调节，或 None 选择自动
    seq_chunk=12,                    # 设置多帧并行计算
)

# bgr = torch.tensor([.47,1,.6]).view(3,1,1).to(DEVICE)
# rec = [None]*4
# downsample_ratio = 0.25

# src = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/3d_image/AVTR1_01_Jakewaking_3D/le/image/AVTAR_01_Jakewaking_00000001.tiff'
# image = cv2.imread(src)
# image = torch.Tensor(image)

# image = image.unsqueeze(0).permute(0,3,1,2).unsqueeze(0)
# with torch.no_grad():
#     fgr,pha,*rec = model(image.to(DEVICE),*rec,downsample_ratio)
#     com = fgr*pha +bgr*(1-pha)

# # torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# def show(image):
#     image = image[0,0].permute(1,2,0).cpu().detach().numpy()
#     plt.imshow(image)
