'''
Author: Qing Hong
Date: 2022-05-25 17:35:01
LastEditors: QingHong
LastEditTime: 2022-06-06 11:25:53
Description: file content
'''
import os,sys
import configparser
# from myutil import mkdir
cur_path = sys.argv[0][:-sys.argv[0][::-1].find('/')]
if 'site-package' in cur_path:
    cur_path = ''
elif cur_path.lower() in ['m','s']:
    cur_path = ''
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)

assert len(sys.argv)>1 ,'please specify config, usage: python start.py your_config'
config_file = sys.argv[1]
# config_file = 'config'
##载入config文件
config = configparser.ConfigParser()
print(cur_path)
config.read(cur_path + config_file, encoding="utf-8")
gpu = config.get('opticalflow','gpu')
gpu = gpu.rstrip().split(',')
num_gpu = len(gpu)
cmd = 'python -m torch.distributed.launch --nproc_per_node={} --use_env main.py {}'.format(num_gpu,config_file)
print(cmd)
output = config.get('opticalflow','output')
mkdir(output)
os.system(cmd)
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py
