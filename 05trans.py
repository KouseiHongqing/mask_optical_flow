import cv2
import sys
import os
import imageio

def half_trans(image_path,save_path):
    image = imageio.imread(image)
    image[...,3] *= 0.5
    save_path = output+'/'+seq+'/'+append+'/mvd_'+re.findall(r'\d+', (f[i].split('/')[-1]))[-1]+ '.exr'
            imageio.imwrite( save_path,ff.astype("float32"))

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    assert len(sys.argv)>2, '输入有问题,请输入输入地址和(+空格)输出地址'
    inp = sys.argv[1]
    output = sys.argv[2]

    assert inp and output , '输入有问题,请输入输入地址和(+空格)输出地址'

    limit = 0
    if len(sys.argv)>3:
        limit = int(sys.argv[3])

    mkdir(output)
    listdir = []
    for i in sorted(os.listdir(inp)):
        if '.png' or '.exr' in i:
            listdir.append(os.path.join(inp,i))
    if limit>0:
        listdir = listdir[:limit]
    



