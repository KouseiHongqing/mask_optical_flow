import cv2
import os

root = '/home/rg0775/QingHong/dataset/guohua_datasets/2d/guohua_dataset/toQing'
inpfile = os.listdir(root)


#获得视频的格式

videoCapture = cv2.VideoCapture(r'/home/rg0775/QingHong/dataset/gma_datasets/ghchen/video_output/4d/result_00005_to_00009.mp4')

#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print('total frames:{}'.format(fNUMS))
def appendzero(a,length=6):
    res = str(a)
    while(len(res)<length):
        res = '0'+ res
    return res
def mkdir(pth):
    if  not os.path.exists(pth):
        os.makedirs(pth)
save_root = '/home/rg0775/QingHong/dataset/gma_datasets/ghchen/video_output/4d/o'
##log
mkdir(save_root)
#读帧
success, frame = videoCapture.read()
index = 0
while success :
    # cv2.imshow('windows', frame) #显示
    # cv2.waitKey(1000//int(fps)) #延迟
    cv2.imwrite(save_root+'/mv'+appendzero(index)+'.png',frame)
    index += 1
    if index %100 ==0:
        print(index)
    success, frame = videoCapture.read() #获取下一帧
 
videoCapture.release()
print('finished')