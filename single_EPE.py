'''
Author: Qing Hong
Date: 2022-06-01 02:56:19
LastEditors: QingHong
LastEditTime: 2022-06-21 14:13:46
Description: file content
'''
from EPE import flow_kitti_mask_error
import os,sys
import imageio
import numpy as np
from tqdm import tqdm

def single_epe_cal(source_file_path,gt_file_path,mask_file_path):
    source_file = sorted(list(filter(lambda x:x[0]!='.',os.listdir(source_file_path))))
    gt_file  = sorted(list(filter(lambda x:x[0]!='.',os.listdir(gt_file_path))))
    mask_file = None if not mask_file_path else sorted(list(filter(lambda x:x[0]!='.',os.listdir(mask_file_path))))
    total_epe = []
    total_acc = []
    for i in tqdm(range(len(source_file))):
        source_ = os.path.join(source_file_path,source_file[i])
        gt_ = os.path.join(gt_file_path,gt_file[i])
        source = imageio.imread(source_)
        source[...,0] *= source.shape[1]
        source[...,1] *= -source.shape[0]
        gt = imageio.imread(gt_)
        gt[...,0] *= gt.shape[1]
        gt[...,1] *= -gt.shape[0]
        if mask_file:
            mask_ = os.path.join(mask_file_path,mask_file[i])
            mask = imageio.imread(mask_)[...,0]
        else:
            mask = np.ones((source.shape[0],source.shape[1]))
        epe = flow_kitti_mask_error(source[...,0],source[...,1],mask,gt[...,0],gt[...,1],mask)
        total_epe.append(epe[0])
        total_acc.append(epe[1])
    print("mean epe:{:.3f}, mean accuracy:{:.3f}%".format(np.array(total_epe).mean(),np.array(total_acc).mean()*100))
    return total_epe,total_acc

def multi_epe_cal(source_file_root,output_root,output_name = 'pca_flow_Char_mv0',image_name='video',gt_name='mv0',mask_name='mask'):
    output_dict = {}
    for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(output_root)))):
        output_dict[i] = os.path.join(output_root,i)
    assert len(output_dict)>0,'no output files!'
    source_dict = {}
    for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(source_file_root)))):
        source_dict[i] = os.path.join(source_file_root,i)
    
    res = {}
    for key,value in output_dict.items():
        source = os.path.join(value,output_name)
        gt = os.path.join(source_dict[key],gt_name)
        mask = os.path.join(source_dict[key],mask_name)
        total_epe,total_acc = single_epe_cal(source,gt,mask)
        res[key] = [np.array(total_epe).mean(),np.array(total_acc).mean()*100]
    return res

def multi_demo():
    source_file_root = '/Users/qhong/Desktop/opt_test_datasets/optical_test_pattern'
    output_root = '/Users/qhong/Desktop/opt_test_datasets/output'
    res = multi_epe_cal(source_file_root,output_root)
    

def single_demo():
    source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_left_original_mv0'
    gt_file_path = '/Users/qhong/Desktop/inpu/scene01/mv0'
    mask_file_path = '/Users/qhong/Desktop/inpu/scene01/mask'
    total_epe,total_acc = single_epe_cal(source_file_path,gt_file_path,mask_file_path)

source_file_root = '/Users/qhong/Desktop/opt_test_datasets/optical_test_pattern'
# output_root = '/Users/qhong/Desktop/opt_test_datasets/output_full_search'
# qwe = ['deepflow_left_original_mv0','farneback_left_original_mv0','rlof_left_original_mv0','simpleflow_left_original_mv0','sparse_to_dense_flow_left_original_mv0']
# output_root = '/Users/qhong/Desktop/opt_test_datasets/output_bounding_box'
# qwe = ['deepflow_Char_mv0','farneback_Char_mv0','rlof_Char_mv0','simpleflow_Char_mv0','sparse_to_dense_flow_Char_mv0','pca_flow_Char_mv0']
output_root = '/Users/qhong/Desktop/opt_test_datasets/output'
qwe = ['deepflow_Char_mv0','farneback_Char_mv0','rlof_Char_mv0','simpleflow_Char_mv0','sparse_to_dense_flow_Char_mv0','pca_flow_Char_mv0']
res = {}
for q in qwe:
    tmp = multi_epe_cal(source_file_root,output_root,output_name=q)
    res[q] = tmp
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_low'
# mean epe:49.289, mean accuracy:20.567% times:107.76
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_medium'
# mean epe:49.095, mean accuracy:20.642% times:179.53
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_high'
# mean epe:47.803, mean accuracy:21.401% times:282.20
# source_file_path = '/Users/qhong/Desktop/trans_output/mv0_veryhigh'
# mean epe:29.149, mean accuracy:22.031% times:339.28

## source
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_left_original_mv0'
# mean epe:4.796, mean accuracy:68.750% times:289.35  mean epe:3.252, mean accuracy:86.734%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_left_original_mv0'
# mean epe:2.224, mean accuracy:83.170% times:2287.24 mean epe:3.032, mean accuracy:89.008%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_left_original_mv0'
# mean epe:8.459, mean accuracy:61.056% times:387.99  mean epe:3.875, mean accuracy:81.222%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_left_original_mv0'
# mean epe:4.406, mean accuracy:76.594% times:235.61  mean epe:3.981, mean accuracy:85.945%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_left_original_mv0'
# mean epe:4.462, mean accuracy:70.988% times:235.14  mean epe:4.565, mean accuracy:72.287%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_left_original_mv0'
# mean epe:3.814, mean accuracy:72.811% times:613.14  mean epe:4.943, mean accuracy:71.212%
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_left_original_mv0'
# mean epe:1.274, mean accuracy:93.654% 1153.8s 

##CF with full
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_Char_CFFULL_mv0'
# mean epe:15.551, mean accuracy:43.260% 306.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_Char_CFFULL_mv0'
# mean epe:3.446, mean accuracy:85.850% 774.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_Char_CFFULL_mv0'
# mean epe:3.992, mean accuracy:81.303% 1079.4s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_Char_CFFULL_mv0'
# mean epe:14.058, mean accuracy:67.757% 91.5s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_Char_CFFULL_mv0'
# mean epe:6.208, mean accuracy:52.960% 67.0s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_Char_CFFULL_mv0'
# mean epe:7.748, mean accuracy:45.492% 56.27s
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_Char_CFFULL_mv0'
# mean epe:6.089, mean accuracy:82.206% 1127.4s

##Char and Bg
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/farneback_Char_mv0'
# mean epe:3.284, mean accuracy:87.212% cost times:28.32s, speed: 5.30
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/deepflow_Char_mv0'
# mean epe:3.446, mean accuracy:85.850% cost times:343.97s, speed: 0.44
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/simpleflow_Char_mv0'
# mean epe:3.594, mean accuracy:82.856% cost times:333.15s, speed: 0.45
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/sparse_to_dense_flow_Char_mv0'
# mean epe:4.171, mean accuracy:81.794% cost times:8.45s, speed: 17.75
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/pca_flow_Char_mv0'
# mean epe:3.290, mean accuracy:83.105% cost times:6.82s, speed: 21.99
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/rlof_Char_mv0'
# mean epe:5.054, mean accuracy:71.783% cost times:60.35s, speed: 2.49
# source_file_path = '/Users/qhong/Desktop/myoutput/scene01/gma_Char_mv0'
# mean epe:2.881, mean accuracy:86.988% cost times:3850.64s, speed: 0.04

# source_files = []


# scene = ['city','sequence:garden_01_30','sequence:garden_02_30:','sequence:garden_03_30','sequence:garden_04_30','sequence:tps_#10_30fps_150','sequence:tps_#10_zp','sequence:tps_#1_30fps_150','sequence:tps_#2_30fps_150','sequence:tps_#3_30fps_150:','sequence:tps_#4_30fps_150:','sequence:tps_#5_30fps_150','sequence:tps_#6_30fps_150','sequence:tps_#7_30fps_150','sequence:tps_#8_30fps_150','sequence:tps_#9_30fps_150','sequence:tps_npc_#2']
# speed = [19.39,20.38,21.93,23.08,24.85,23.99,23.06,22.18,23.17,21.30,19.46,22.40,23.20,23.46,23.48,23.10,23.19]
# speed = [7.16,6.1,7.66,6.99,6.19,8.26,8.09,6.65,6.46,7.55,7.78,7.14,8.15,8.95,6.92,7.06,6.12]

# pca_flow_mask = {'city': [6.2445693, 55.769580936920995],
#  'garden_01_30': [0.27358493, 99.952353662352],
#  'garden_02_30': [1.0540143, 94.95182315754779],
#  'garden_03_30': [3.9814157, 72.12855958633233],
#  'garden_04_30': [2.0862727, 84.35754717090624],
#  'tps_#10_30fps_150': [0.14571463, 99.56979642757308],
#  'tps_#10_zp': [0.09215464, 99.75570316120123],
#  'tps_#1_30fps_150': [1.2540404, 88.9129983546838],
#  'tps_#2_30fps_150': [4.6233134, 67.24571677701492],
#  'tps_#3_30fps_150': [0.06288174, 99.99386205364812],
#  'tps_#4_30fps_150': [1.6338521, 89.34365000837913],
#  'tps_#5_30fps_150': [1.4844142, 89.90104607721563],
#  'tps_#6_30fps_150': [3.3088756, 81.25227381764267],
#  'tps_#7_30fps_150': [4.729024, 78.05154503390875],
#  'tps_#8_30fps_150': [1.2980767, 91.25564658647494],
#  'tps_#9_30fps_150': [0.88513356, 93.62168057430253],
#  'tps_npc_#2': [1.3555058, 88.67421815592516]}

# pca_flow_full = {'city': [9.061729, 36.26434468021847],
#  'garden_01_30': [1.2504537, 90.26478341320214],
#  'garden_02_30': [2.3420467, 73.83327707527368],
#  'garden_03_30': [4.8169484, 62.21545605242259],
#  'garden_04_30': [4.1385374, 63.85766627230128],
#  'tps_#10_30fps_150': [1.6207268, 87.39471379388887],
#  'tps_#10_zp': [1.9718437, 83.80252039385608],
#  'tps_#1_30fps_150': [2.3964589, 74.946273421804],
#  'tps_#2_30fps_150': [5.1372447, 61.312573855390205],
#  'tps_#3_30fps_150': [1.3317498, 92.04443375683142],
#  'tps_#4_30fps_150': [2.767204, 78.95280618794473],
#  'tps_#5_30fps_150': [2.6768982, 77.46049358662226],
#  'tps_#6_30fps_150': [3.7586505, 74.92965159637107],
#  'tps_#7_30fps_150': [5.5499997, 67.9324543925064],
#  'tps_#8_30fps_150': [2.8962395, 71.91760772708295],
#  'tps_#9_30fps_150': [1.7172147, 86.7920991921403],
#  'tps_npc_#2': [2.2069297, 79.44635350611891]}



# ##mask search speed:
# deepflow_mask=[0.45]
# farneback=[5.24,5.16,5.15,5.13,5.18,5.19,5.20,5.20,5.16,5.23,5.21,5.21,5.21,5.16,5.30,5.19]
# simpleflow=[0.45]
# sparse_to_dense_flow=[16.81,17.08,16.63,17.19,16.95,16.89,17.62,16.49,17.23,15.73,16.91,16.78,17.05,17.16,17.24,16.90,17.16]
# rlof=[2.66]


# {'deepflow_Char_mv0': {'city': [5.5946555, 68.98928489846995],
#   'garden_01_30': [0.15118206, 99.94784550043163],
#   'garden_02_30': [0.69223076, 95.03861682814622],
#   'garden_03_30': [3.4917815, 77.48512808193675],
#   'garden_04_30': [1.6092048, 88.83003387111923],
#   'tps_#10_30fps_150': [0.1212927, 99.56979642757308],
#   'tps_#10_zp': [0.058060743, 99.77525632216766],
#   'tps_#1_30fps_150': [0.9264039, 91.29165048135378],
#   'tps_#2_30fps_150': [4.0400286, 75.08836633982855],
#   'tps_#3_30fps_150': [0.03867398, 99.96638710773],
#   'tps_#4_30fps_150': [1.1266901, 93.65463216373394],
#   'tps_#5_30fps_150': [0.9295644, 94.35121106855594],
#   'tps_#6_30fps_150': [3.0229297, 83.67945840492735],
#   'tps_#7_30fps_150': [4.265417, 82.94985407388123],
#   'tps_#8_30fps_150': [0.85765505, 95.07139342603294],
#   'tps_#9_30fps_150': [0.63472474, 95.46036838091845],
#   'tps_npc_#2': [0.99532336, 91.6415703116081]},
#  'farneback_Char_mv0': {'city': [6.1811504, 65.34779862791213],
#   'garden_01_30': [0.46102768, 98.83433994564948],
#   'garden_02_30': [1.4366622, 86.80780797130575],
#   'garden_03_30': [3.754585, 75.22938136773689],
#   'garden_04_30': [1.8288409, 87.06826265600203],
#   'tps_#10_30fps_150': [0.23695092, 99.41195480529653],
#   'tps_#10_zp': [0.19551763, 99.17445031256568],
#   'tps_#1_30fps_150': [1.5125222, 86.77770399877551],
#   'tps_#2_30fps_150': [5.605168, 62.85013879030294],
#   'tps_#3_30fps_150': [0.0704511, 99.81129412820373],
#   'tps_#4_30fps_150': [1.6109204, 90.13251958765677],
#   'tps_#5_30fps_150': [1.4871383, 90.30672850803937],
#   'tps_#6_30fps_150': [3.9864514, 73.88468996127392],
#   'tps_#7_30fps_150': [5.748497, 70.01211879541118],
#   'tps_#8_30fps_150': [1.283113, 91.05666912087949],
#   'tps_#9_30fps_150': [1.7495232, 84.53642892205113],
#   'tps_npc_#2': [1.7544773, 83.74986828297314]},
#  'rlof_Char_mv0': {'city': [8.980083, 49.68784738594923],
#   'garden_01_30': [1.31527, 89.88746206584],
#   'garden_02_30': [2.1503582, 76.61842217533507],
#   'garden_03_30': [4.929405, 62.37785869952186],
#   'garden_04_30': [3.3201509, 71.33989793731934],
#   'tps_#10_30fps_150': [0.8687473, 95.56920108120137],
#   'tps_#10_zp': [0.79960114, 95.67968898662068],
#   'tps_#1_30fps_150': [1.8644056, 83.19603126514667],
#   'tps_#2_30fps_150': [5.5448837, 63.815659305248815],
#   'tps_#3_30fps_150': [0.2495494, 99.35321522717557],
#   'tps_#4_30fps_150': [2.6533577, 83.16473329355145],
#   'tps_#5_30fps_150': [2.3895183, 84.55306852089231],
#   'tps_#6_30fps_150': [3.4681082, 79.98308875228679],
#   'tps_#7_30fps_150': [5.3215322, 74.85859957075792],
#   'tps_#8_30fps_150': [2.4774876, 81.62831804833799],
#   'tps_#9_30fps_150': [1.1059285, 92.75752638582242],
#   'tps_npc_#2': [1.8616918, 84.61242351956135]},
#  'simpleflow_Char_mv0': {'city': [8.057071, 56.44925018394079],
#   'garden_01_30': [0.29054293, 99.97343916423922],
#   'garden_02_30': [0.93051845, 93.29130499494185],
#   'garden_03_30': [3.855038, 74.7751669084157],
#   'garden_04_30': [2.121222, 85.10334276478193],
#   'tps_#10_30fps_150': [0.22236705, 99.3108409007766],
#   'tps_#10_zp': [0.11589731, 99.78381570658021],
#   'tps_#1_30fps_150': [1.2290052, 89.63725594825598],
#   'tps_#2_30fps_150': [4.786905, 71.12532894627819],
#   'tps_#3_30fps_150': [0.09923591, 99.56936190677162],
#   'tps_#4_30fps_150': [1.8584148, 89.43529199061881],
#   'tps_#5_30fps_150': [1.587695, 90.64959836846785],
#   'tps_#6_30fps_150': [3.3036466, 81.17454407945225],
#   'tps_#7_30fps_150': [5.814833, 73.19793159797857],
#   'tps_#8_30fps_150': [1.6436604, 90.21839759506153],
#   'tps_#9_30fps_150': [0.77555877, 96.21727669078508],
#   'tps_npc_#2': [1.3250039, 90.02889266030658]},
#  'sparse_to_dense_flow_Char_mv0': {'city': [7.7000284, 60.2298386548608],
#   'garden_01_30': [0.2878496, 99.62146659039026],
#   'garden_02_30': [1.3275692, 87.69925297263399],
#   'garden_03_30': [4.4159517, 69.35880978493752],
#   'garden_04_30': [2.1306908, 83.71727091212723],
#   'tps_#10_30fps_150': [0.1424581, 99.56693879007817],
#   'tps_#10_zp': [0.06907842, 99.69237207466183],
#   'tps_#1_30fps_150': [1.1598089, 89.7195949118416],
#   'tps_#2_30fps_150': [4.8110313, 71.26787195536227],
#   'tps_#3_30fps_150': [0.02210663, 99.99386205364812],
#   'tps_#4_30fps_150': [1.7220473, 91.1991711383099],
#   'tps_#5_30fps_150': [1.5038618, 91.75440592780872],
#   'tps_#6_30fps_150': [3.2139485, 82.52870087863937],
#   'tps_#7_30fps_150': [4.6907625, 80.15476228951248],
#   'tps_#8_30fps_150': [1.2388287, 93.17014743409094],
#   'tps_#9_30fps_150': [0.7786848, 94.95852044259365],
#   'tps_npc_#2': [1.308502, 89.07626184289866]},
#  'pca_flow_Char_mv0': {'city': [6.2445693, 55.769580936920995],
#   'garden_01_30': [0.27358493, 99.952353662352],
#   'garden_02_30': [1.0540143, 94.95182315754779],
#   'garden_03_30': [3.9814157, 72.12855958633233],
#   'garden_04_30': [2.0862727, 84.35754717090624],
#   'tps_#10_30fps_150': [0.14571463, 99.56979642757308],
#   'tps_#10_zp': [0.09215464, 99.75570316120123],
#   'tps_#1_30fps_150': [1.2540404, 88.9129983546838],
#   'tps_#2_30fps_150': [4.6233134, 67.24571677701492],
#   'tps_#3_30fps_150': [0.06288174, 99.99386205364812],
#   'tps_#4_30fps_150': [1.6338521, 89.34365000837913],
#   'tps_#5_30fps_150': [1.4844142, 89.90104607721563],
#   'tps_#6_30fps_150': [3.3088756, 81.25227381764267],
#   'tps_#7_30fps_150': [4.729024, 78.05154503390875],
#   'tps_#8_30fps_150': [1.2980767, 91.25564658647494],
#   'tps_#9_30fps_150': [0.88513356, 93.62168057430253],
#   'tps_npc_#2': [1.3555058, 88.67421815592516]}}

{'deepflow_Char_mv0': [5.59, 0.15, 0.69, 3.49, 1.61, 0.12, 0.06, 0.93, 4.04, 0.04, 1.13, 0.93, 3.02, 4.27, 0.86, 0.63, 1.0], 'farneback_Char_mv0': [6.18, 0.46, 1.44, 3.75, 1.83, 0.24, 0.2, 1.51, 5.61, 0.07, 1.61, 1.49, 3.99, 5.75, 1.28, 1.75, 1.75], 'rlof_Char_mv0': [8.98, 1.32, 2.15, 4.93, 3.32, 0.87, 0.8, 1.86, 5.54, 0.25, 2.65, 2.39, 3.47, 5.32, 2.48, 1.11, 1.86], 'simpleflow_Char_mv0': [8.06, 0.29, 0.93, 3.86, 2.12, 0.22, 0.12, 1.23, 4.79, 0.1, 1.86, 1.59, 3.3, 5.81, 1.64, 0.78, 1.33], 'sparse_to_dense_flow_Char_mv0': [7.7, 0.29, 1.33, 4.42, 2.13, 0.14, 0.07, 1.16, 4.81, 0.02, 1.72, 1.5, 3.21, 4.69, 1.24, 0.78, 1.31], 'pca_flow_Char_mv0': [6.24, 0.27, 1.05, 3.98, 2.09, 0.15, 0.09, 1.25, 4.62, 0.06, 1.63, 1.48, 3.31, 4.73, 1.3, 0.89, 1.36]}

{'deepflow_Char_mv0': [68.99, 99.95, 95.04, 77.49, 88.83, 99.57, 99.78, 91.29, 75.09, 99.97, 93.65, 94.35, 83.68, 82.95, 95.07, 95.46, 91.64], 'farneback_Char_mv0': [65.35, 98.83, 86.81, 75.23, 87.07, 99.41, 99.17, 86.78, 62.85, 99.81, 90.13, 90.31, 73.88, 70.01, 91.06, 84.54, 83.75], 'rlof_Char_mv0': [49.69, 89.89, 76.62, 62.38, 71.34, 95.57, 95.68, 83.2, 63.82, 99.35, 83.16, 84.55, 79.98, 74.86, 81.63, 92.76, 84.61], 'simpleflow_Char_mv0': [56.45, 99.97, 93.29, 74.78, 85.1, 99.31, 99.78, 89.64, 71.13, 99.57, 89.44, 90.65, 81.17, 73.2, 90.22, 96.22, 90.03], 'sparse_to_dense_flow_Char_mv0': [60.23, 99.62, 87.7, 69.36, 83.72, 99.57, 99.69, 89.72, 71.27, 99.99, 91.2, 91.75, 82.53, 80.15, 93.17, 94.96, 89.08], 'pca_flow_Char_mv0': [55.77, 99.95, 94.95, 72.13, 84.36, 99.57, 99.76, 88.91, 67.25, 99.99, 89.34, 89.9, 81.25, 78.05, 91.26, 93.62, 88.67]}

# ##fully search
# farneback=[5.01,5.12,5.10,5.07,5.22,2.13,5.16,5.19,5.20,5.17,5.13,5.15,5.21,5.17,5.20,5.09,5.09]
# deep_flow = [0.45]
# simpleflow=[0.45]
# sparse_to_dense_flow=[5.52,6.13,5.65,5.93,5.80,5.36,5.40,5.51,5.90,5.56,5.33,5.56,6.24,6.32,5.63,5.45,5.94]
# {'deepflow_left_original_mv0': {'city': [7.183387, 63.33976381503072],
#   'garden_01_30': [0.220045, 99.14162812328716],
#   'garden_02_30': [1.0170071, 92.07894752760262],
#   'garden_03_30': [3.7725134, 75.67093459818808],
#   'garden_04_30': [2.908753, 82.77115678937518],
#   'tps_#10_30fps_150': [0.3960271, 98.15791698857002],
#   'tps_#10_zp': [0.49976158, 97.19095559790969],
#   'tps_#1_30fps_150': [1.1227211, 89.70796408005303],
#   'tps_#2_30fps_150': [4.248266, 73.23326657215804],
#   'tps_#3_30fps_150': [0.40796763, 98.29641263473383],
#   'tps_#4_30fps_150': [1.6838484, 91.07974945038632],
#   'tps_#5_30fps_150': [1.6039599, 91.56354835944876],
#   'tps_#6_30fps_150': [3.0423994, 81.40484362006202],
#   'tps_#7_30fps_150': [5.014112, 79.52112525762904],
#   'tps_#8_30fps_150': [1.3652469, 92.79575778121512],
#   'tps_#9_30fps_150': [0.78683597, 94.08497039573899],
#   'tps_npc_#2': [1.1890086, 90.36162262838707]},
#  'farneback_left_original_mv0': {'city': [7.24247, 59.274430214563935],
#   'garden_01_30': [0.65871066, 96.25276009983035],
#   'garden_02_30': [1.6534383, 85.20525186012387],
#   'garden_03_30': [3.9678934, 72.73684737438005],
#   'garden_04_30': [2.3121777, 83.8049462414507],
#   'tps_#10_30fps_150': [0.6785923, 96.17581418679906],
#   'tps_#10_zp': [1.0302894, 94.43112008374813],
#   'tps_#1_30fps_150': [1.630707, 85.26301151026698],
#   'tps_#2_30fps_150': [5.4950624, 62.27952085476838],
#   'tps_#3_30fps_150': [0.5703935, 95.79142725827101],
#   'tps_#4_30fps_150': [1.8373759, 88.69562982018748],
#   'tps_#5_30fps_150': [1.6250588, 89.68698540438231],
#   'tps_#6_30fps_150': [4.1185937, 71.37478754061318],
#   'tps_#7_30fps_150': [6.040458, 68.02824483978715],
#   'tps_#8_30fps_150': [1.4851234, 89.79822843267878],
#   'tps_#9_30fps_150': [2.018498, 82.01434556111968],
#   'tps_npc_#2': [1.8660307, 82.22040344705022]},
#  'rlof_left_original_mv0': {'city': [9.28847, 46.335916163378535],
#   'garden_01_30': [1.4895145, 87.82014036893924],
#   'garden_02_30': [2.3718708, 74.45011832156212],
#   'garden_03_30': [5.1076517, 60.00210633217436],
#   'garden_04_30': [3.803421, 66.48882418952617],
#   'tps_#10_30fps_150': [1.1525271, 93.81034310058183],
#   'tps_#10_zp': [1.0852873, 93.75805868512607],
#   'tps_#1_30fps_150': [2.0887036, 81.15737712952057],
#   'tps_#2_30fps_150': [5.4338803, 63.59744951237407],
#   'tps_#3_30fps_150': [0.57470274, 97.13641764192762],
#   'tps_#4_30fps_150': [2.7404647, 82.19431465243674],
#   'tps_#5_30fps_150': [2.5854173, 83.13211072221104],
#   'tps_#6_30fps_150': [3.4147646, 79.17018327902852],
#   'tps_#7_30fps_150': [5.411172, 73.59615781382965],
#   'tps_#8_30fps_150': [2.6954558, 79.78404046252508],
#   'tps_#9_30fps_150': [1.2530335, 91.50543936833787],
#   'tps_npc_#2': [2.0490608, 82.43881762492802]},
#  'simpleflow_left_original_mv0': {'city': [8.389053, 55.07601871673904],
#   'garden_01_30': [0.3295206, 99.66691463786506],
#   'garden_02_30': [0.965254, 92.74190222641549],
#   'garden_03_30': [3.9882538, 73.71115164659263],
#   'garden_04_30': [2.5106592, 82.33463608584258],
#   'tps_#10_30fps_150': [0.23464489, 99.28502806312969],
#   'tps_#10_zp': [0.12943138, 99.71575866752745],
#   'tps_#1_30fps_150': [1.2814792, 89.39887460394172],
#   'tps_#2_30fps_150': [4.8413815, 70.92009536471173],
#   'tps_#3_30fps_150': [0.10882384, 99.55048929120899],
#   'tps_#4_30fps_150': [1.965688, 88.85597735635413],
#   'tps_#5_30fps_150': [1.7320329, 89.60472004380948],
#   'tps_#6_30fps_150': [3.3195634, 80.92261632707576],
#   'tps_#7_30fps_150': [5.955754, 72.49767689898248],
#   'tps_#8_30fps_150': [1.7454666, 89.51164601378933],
#   'tps_#9_30fps_150': [0.7933809, 96.07806043635297],
#   'tps_npc_#2': [1.418868, 89.35376407342851]},
#  'sparse_to_dense_flow_left_original_mv0': {'city': [7.3929067,
#    59.63310643968091],
#   'garden_01_30': [0.5677898, 96.82549014997446],
#   'garden_02_30': [1.5795386, 85.48940916029746],
#   'garden_03_30': [4.2103057, 71.29210875871216],
#   'garden_04_30': [2.3949893, 83.54731878478162],
#   'tps_#10_30fps_150': [0.3585951, 98.71423629862743],
#   'tps_#10_zp': [0.3919691, 98.4884741233452],
#   'tps_#1_30fps_150': [1.3124373, 88.78944878928672],
#   'tps_#2_30fps_150': [4.9779243, 70.2282016059746],
#   'tps_#3_30fps_150': [0.13744932, 99.44480248962768],
#   'tps_#4_30fps_150': [1.6753231, 91.2041262949201],
#   'tps_#5_30fps_150': [1.4389256, 91.55286511323007],
#   'tps_#6_30fps_150': [3.4643333, 80.21820139419064],
#   'tps_#7_30fps_150': [5.492541, 77.18414512728118],
#   'tps_#8_30fps_150': [1.3297566, 92.6679506430087],
#   'tps_#9_30fps_150': [0.97601235, 93.34378876909064],
#   'tps_npc_#2': [1.371589, 88.49787479543227]}}


# {'deepflow_left_original_mv0': [7.18, 0.22, 1.02, 3.77, 2.91, 0.4, 0.5, 1.12, 4.25, 0.41, 1.68, 1.6, 3.04, 5.01, 1.37, 0.79, 1.19], 'farneback_left_original_mv0': [7.24, 0.66, 1.65, 3.97, 2.31, 0.68, 1.03, 1.63, 5.5, 0.57, 1.84, 1.63, 4.12, 6.04, 1.49, 2.02, 1.87], 'rlof_left_original_mv0': [9.29, 1.49, 2.37, 5.11, 3.8, 1.15, 1.09, 2.09, 5.43, 0.57, 2.74, 2.59, 3.41, 5.41, 2.7, 1.25, 2.05], 'simpleflow_left_original_mv0': [8.39, 0.33, 0.97, 3.99, 2.51, 0.23, 0.13, 1.28, 4.84, 0.11, 1.97, 1.73, 3.32, 5.96, 1.75, 0.79, 1.42], 'sparse_to_dense_flow_left_original_mv0': [7.39, 0.57, 1.58, 4.21, 2.39, 0.36, 0.39, 1.31, 4.98, 0.14, 1.68, 1.44, 3.46, 5.49, 1.33, 0.98, 1.37]}

# {'deepflow_left_original_mv0': [63.34, 99.14, 92.08, 75.67, 82.77, 98.16, 97.19, 89.71, 73.23, 98.3, 91.08, 91.56, 81.4, 79.52, 92.8, 94.08, 90.36], 'farneback_left_original_mv0': [59.27, 96.25, 85.21, 72.74, 83.8, 96.18, 94.43, 85.26, 62.28, 95.79, 88.7, 89.69, 71.37, 68.03, 89.8, 82.01, 82.22], 'rlof_left_original_mv0': [46.34, 87.82, 74.45, 60.0, 66.49, 93.81, 93.76, 81.16, 63.6, 97.14, 82.19, 83.13, 79.17, 73.6, 79.78, 91.51, 82.44], 'simpleflow_left_original_mv0': [55.08, 99.67, 92.74, 73.71, 82.33, 99.29, 99.72, 89.4, 70.92, 99.55, 88.86, 89.6, 80.92, 72.5, 89.51, 96.08, 89.35], 'sparse_to_dense_flow_left_original_mv0': [59.63, 96.83, 85.49, 71.29, 83.55, 98.71, 98.49, 88.79, 70.23, 99.44, 91.2, 91.55, 80.22, 77.18, 92.67, 93.34, 88.5]}

# ##bounding box
# deep_flow = [3.26,3.70,3.69,4.37,5.18,4.11,4.10,3.63,3.90,2.22,3.76,3.88,3.89,3.75,3.80,3.36,3.90]
# simpleflow = [5.12, 6.1, 5.85, 7.99, 11.15, 7.01, 6.4, 5.6, 6.2, 2.5, 6.32, 6.63, 7.11, 6.48, 6.72, 5.46, 7.12]
# sparse_to_dense_flow = [46.74, 59.63, 67.56, 84.85, 104.36, 83.55, 82.27, 70.34, 70.96, 46.12, 76.98, 74.64, 76.95, 72.67, 75.33, 64.21, 75.88]
# rlof = [19.14, 21.42, 19.92, 31.58, 35.82, 34.14, 34.03, 23.4, 26.07, 9.94, 23.85, 28.65, 29.99, 27.36, 24.29, 20.01, 29.26]
# pca_flow = [42.61, 48.15, 50.41, 54.13, 66.06, 49.16, 47.8, 50.11, 52.75, 31.71, 50.77, 51.21, 51.45, 52.05, 50.4, 46.96, 51.75]
# farneback = [61.12, 73.92, 72.82, 97.29, 138.76, 83.15, 80.08, 72.71, 78.88, 30.66, 74.95, 77.9, 82.86, 78.8, 80.23, 71.0, 79.63]


# 'deepflow_Char_mv0': [8.27, 0.15, 0.68, 4.45, 3.33, 0.12, 0.06, 0.96, 4.44, 0.04, 1.39, 1.17, 3.0, 4.9, 1.05, 0.63, 1.0], 'farneback_Char_mv0': [13.92, 0.32, 1.64, 5.32, 5.77, 0.24, 0.2, 1.71, 6.29, 0.07, 2.87, 2.88, 4.01, 7.34, 2.21, 1.6, 2.25], 'rlof_Char_mv0': [11.47, 0.37, 1.39, 4.68, 4.17, 0.22, 0.16, 1.94, 5.77, 0.06, 2.8, 2.84, 3.61, 5.57, 2.5, 0.96, 2.02], 'simpleflow_Char_mv0': [14.09, 0.29, 1.17, 5.04, 4.86, 0.22, 0.12, 1.36, 5.02, 0.1, 2.39, 2.38, 3.36, 6.31, 2.05, 0.78, 1.63], 'sparse_to_dense_flow_Char_mv0': [8.07, 0.29, 1.32, 4.24, 2.58, 0.14, 0.07, 1.15, 4.89, 0.02, 1.69, 1.37, 3.23, 4.58, 1.21, 0.78, 1.26], 'pca_flow_Char_mv0': [7.13, 0.22, 0.87, 3.87, 2.53, 0.15, 0.11, 1.25, 4.74, 0.05, 1.75, 1.61, 3.25, 5.08, 1.33, 0.83, 1.33]

# {'deepflow_Char_mv0': [62.35, 99.95, 95.08, 74.85, 82.99, 99.57, 99.78, 91.06, 73.75, 99.97, 92.66, 93.43, 83.82, 81.36, 94.5, 95.5, 91.7], 'farneback_Char_mv0': [44.91, 99.35, 85.09, 71.45, 69.43, 99.41, 99.17, 85.02, 60.66, 99.83, 85.04, 84.97, 73.14, 63.59, 87.98, 87.37, 81.15], 'rlof_Char_mv0': [49.43, 99.72, 86.7, 68.56, 76.68, 99.54, 99.6, 80.76, 60.06, 99.99, 80.06, 79.74, 76.36, 71.94, 80.7, 93.22, 80.28], 'simpleflow_Char_mv0': [35.36, 99.97, 91.94, 70.23, 70.99, 99.3, 99.78, 88.55, 70.48, 99.57, 86.24, 86.11, 80.85, 69.04, 87.43, 96.19, 87.37], 'sparse_to_dense_flow_Char_mv0': [60.67, 99.62, 87.35, 69.96, 83.94, 99.57, 99.69, 90.15, 70.87, 99.99, 91.66, 92.42, 82.64, 80.49, 93.62, 94.84, 89.7], 'pca_flow_Char_mv0': [60.21, 99.94, 95.36, 75.04, 83.15, 99.57, 99.7, 88.73, 67.77, 99.99, 90.1, 90.86, 81.71, 77.13, 92.09, 94.09, 89.42]}

def getm(a):
    res1,res2 = {},{}
    for item,value in a.items():
        tmp1,tmp2 = [],[]
        for epe,acc in value.values():
            tmp1.append(round(epe,2))
            tmp2.append(round(acc,2))
        res1[item] = tmp1
        res2[item] = tmp2
    return res1,res2