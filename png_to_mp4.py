'''
Author: Qing Hong
Date: 2022-05-04 12:26:59
LastEditors: QingHong
LastEditTime: 2022-06-15 16:18:42
Description: file content
'''
import cv2
import os
from tqdm import tqdm
import re
from myutil import mkdir

def png2mp4(png_path,save_path,cap_fps=10):
    tmp_path = sorted(list(filter(lambda x:x[0]!='.',os.listdir(png_path))))
    png_image_path = [os.path.join(png_path,i) for i in tmp_path]
    assert len(png_image_path)>0,'no images!'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    # size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），
    # 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
    size = None #size（width，height）
    # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
    # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
    mkdir(save_path)
    save_path += '/' + png_image_path[0].split('/')[-3] +'.mp4'
    video = None
    for i in tqdm(range(len(png_image_path)),desc='processing png to mp4'):
        if not size:
            h,w,_ = cv2.imread(png_image_path[i]).shape
            size = (w,h)
        if not video:
            video = cv2.VideoWriter(save_path, fourcc, cap_fps, size)
        video.write(cv2.imread(png_image_path[i]))
    video.release()

def raw_png2mp4(png_list,save_path,cap_fps=10):
    assert len(png_list)>0,'no images!'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    # size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），
    # 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
    size = None #size（width，height）
    # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
    # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
    video = None
    for i in tqdm(range(len(png_list)),desc='processing png to mp4'):
        if not size:
            h,w,_ = png_list[i].shape
            size = (w,h)
        if not video:
            video = cv2.VideoWriter(save_path, fourcc, cap_fps, size)
        video.write(png_list[i])
    video.release()

png_path = '/Users/qhong/Desktop/opt_test_datasets/output/city/pca_flow_Char_mv0_restore'
save_path = '/Users/qhong/Desktop/opt_test_datasets/output/city/pca_flow_Char_mv0_restore_mp4'
png2mp4(png_path,save_path)