import os,sys
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/core'
sys.path.insert(0, dir_mytest)
from utils.utils import InputPadder
from utils import flow_viz
from network import RAFTGMA
import torch
import argparse


def load_torch_model(args,DEVICE,model_path):
    model=RAFTGMA(args)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    return model

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



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()
net = load_torch_model(args,DEVICE,'/home/rg0775/QingHong/dataset/gma_datasets/mask_optical_flow/checkpoints/gma-sintel.pth')  ##加载torch模型
net.eval()
x = torch.randn(1 ,3 ,224 ,224)  #模型输入数据
y = torch.randn(1 ,3 ,224 ,224)  #模型输入数据
torch_output = net(x,y)


export_onnx_file = '/onnx_file/gma_intel.onnx' #onnx模型存储地址
torch.onnx.export(net,
                    (x,y),
                    export_onnx_file,
                    verbose=True,
                    do_constant_folding=True,
                    opset_version=12)