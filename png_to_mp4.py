import cv2
import os
from tqdm import tqdm
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
# cap_fps是帧率，根据自己需求设置帧率
cap_fps = 60
# size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），

# 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
size = None #size（width，height）
# 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
# video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
video = None
sli = 2
all_data = []
all_data.append([r'/home/rg0775/QingHong/dataset/gma_datasets/ghchen/421output/mori_b/clip01/gma_left_right_mergedepth_mv0',r'/home/rg0775/QingHong/dataset/gma_datasets/ghchen/video_output/4d'])
for path,save_path in all_data:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_lst = sorted(os.listdir(path))
    # 259200 285530
    n = len(file_lst)
    for i in tqdm(range(n)):
        if not size:
            h,w,_ = cv2.imread(path+'/'+file_lst[0]).shape
            size = (w,h)
        if i % round(n/sli) == 0:
            if video:
                video.release()
            fname = file_lst[i][-10:-5]
            fname_ = str(int(fname) + round(n/sli)-1) if i != (round(n/sli))*(sli-1) else file_lst[n-1][-10:-5]
            video = cv2.VideoWriter(os.path.join(save_path,'result_{}_to_{}.mp4'.format(fname,fname_)), fourcc, cap_fps, size)#设置保存视频的名称和路径，默认在根目>录下
        fil = os.path.join(path,file_lst[i])
        img = cv2.imread(fil)
        video.write(img)
# tar cvf image.zip AV1R3/LEFT/Avatar1_R03_P3D65_48nit_4096x2304_LEFT_PNG/ AV1R3/RIGHT/Avatar1_R03_P3D65_48nit_4096x2304_RIGHT_PNG/