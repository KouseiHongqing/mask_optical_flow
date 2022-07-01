import os
import cv2
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)
root = r'/home/rg0775/QingHong/dataset/gma_datasets/mask_optical_flow/datasets/Basketball/image'
first = sorted(os.listdir(root))

cproot = root+'_4k'
mkdir(cproot)

second = []
second_=[]
for i in first:
    if i[0] == '.':
        continue
    second.append(os.path.join(root,i))
    second_.append(os.path.join(cproot,i))


for i in range(len(second)):
        image = cv2.imread(second[i])
        image = cv2.resize(image,(2048,1152))
        cv2.imwrite(second_[i],image)
