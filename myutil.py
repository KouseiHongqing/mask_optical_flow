'''
Author: Qing Hong
Date: 2022-03-08 17:21:31
LastEditors: QingHong
LastEditTime: 2022-06-15 14:23:32
Description: file content
'''
import os
import time
import shutil
import numpy as np

cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)

def build_dir(mk_dir,algo):
    if  not os.path.exists(mk_dir):
        os.makedirs(mk_dir)

    save_dir =  mk_dir + '/' + algo +'/'+ cur_time_sec
    if  os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print('saving file created:{}'.format(save_dir))

import matplotlib.pyplot as plt
def pt(image):
    plt.imshow(image[...,::-1])
def pt2(image):
    plt.imshow(np.insert(image,1,0,axis=2)[...,::-1])
def plot(image):
    plt.imshow(image)
from PIL import Image

def show(image):
    im = Image.fromarray(image.astype('uint8')).convert('RGB')
    im.show()

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def delpoint(root,arr):
        tmp = []
        for ar in arr:
            if ar[0] != '.' and os.path.isdir(root+'/' +ar):
                tmp.append(ar)
        return tmp

def prune_point(file):
    res = []
    for i in file:
        if i[0]!='.':
            res.append(i)
    return res


def imresize(image, ratio=None, out_size=None, method='bicubic', start='auto,auto', out_offset=None, padding='symmetric', clip=True):
    '''
    Parameters
    ----------
    image     : ndarray, 1 channel or n channels interleaved
    ratio     : scale ratio. It can be a scalar, or a list/tuple/numpy.array.
                If it's a scalar, the ratio applies to both H and V.
                If it's a list/numpy.array, it specifies the hor_ratio and ver_ratio.
    out_size  : output size [wo, ho]
    method    : 'bicubic' | 'bilinear' | 'nearest'
    start     : string seperated by ',' specify the start position of x and y
    out_offset: offset at output domain [xoffset, yoffset]
    padding   : 'zeros' | 'edge','replicate','border' | 'symmetric'. Default: 'symmetric' (TBD)
    clip      : only effect for float image data (uint8/uint16 image output is alway clipped)

    Returns
    -------
    result: ndarray

    History
    2021/07/10: changed ratio order [H,W] -> [W,H]
                add out_offset
    2021/07/11: add out_size
    2021/07/31: ratio cannot be used as resolution any more

    Notes：
    如果 ratio 和 out_size 都没有指定，则 ratio = 1
    如果只指定 out_size，则 ratio 按输入图像尺寸和 out_size 计算
    如果只指定 ratio，则输出尺寸为输入图像尺寸和 ratio 的乘积并四舍五入
    如果同时指定 ratio 和 out_size，则按  ratio 输出 out_size 大小的图，这时既保证 ratio，也保证输出图像尺寸
    '''
    startx, starty = start.split(',')
    ih, iw = image.shape[:2]

    if ratio is None:
        ratio = 1 if out_size is None else [out_size[0]/iw, out_size[1]/ih]

    if isinstance(ratio, list) or isinstance(ratio, np.ndarray) or isinstance(ratio, tuple):
        hratio, vratio = ratio[0], ratio[1]
    else:
        hratio, vratio = ratio, ratio

    if out_offset is None: out_offset = (0, 0)
    if out_size   is None: out_size   = (None, None)

    if method == 'bicubic':
        outv = ver_interp_bicubic(image, vratio, out_size[1], starty, out_offset[1], clip)
        out  = hor_interp_bicubic(outv, hratio, out_size[0], startx, out_offset[0], clip)
    else:
        xinc, yinc = 1/hratio, 1/vratio
        ow = round(iw * hratio) if out_size[0] is None else out_size[0]
        oh = round(ih * vratio) if out_size[1] is None else out_size[1]
        x0 = (-.5 + xinc/2 if startx == 'auto' else float(startx)) + out_offset[0] * xinc # (x0, y0) is in input domain
        y0 = (-.5 + yinc/2 if starty == 'auto' else float(starty)) + out_offset[1] * yinc 
        x = x0 + np.arange(ow) * xinc
        y = y0 + np.arange(oh) * yinc
        xaux = np.r_[np.arange(iw), np.arange(iw-1,-1,-1)] # 0, 1, ..., iw-2, iw-1, iw-1, iw-2, ..., 1, 0
        yaux = np.r_[np.arange(ih), np.arange(ih-1,-1,-1)]
        if method == 'nearest':
            x = np.floor(x + .5).astype('int32') # don't use np.round() as it rounds to even value (w,)
            y = np.floor(y + .5).astype('int32')
            xind = xaux[np.mod(np.int32(x), xaux.size)]
            yind = yaux[np.mod(np.int32(y), yaux.size)]
            out = image[np.ix_(yind, xind)]
        elif method == 'bilinear':
            tlx = np.floor(x).astype('int32')
            tly = np.floor(y).astype('int32')
            wy, wx = np.ix_(y - tly, x - tlx) # wy: (h, 1), wx: (1, w)
            brx = xaux[np.mod(tlx + 1, xaux.size)]
            bry = yaux[np.mod(tly + 1, yaux.size)]
            tlx = xaux[np.mod(tlx    , xaux.size)]
            tly = yaux[np.mod(tly    , yaux.size)]
            if image.ndim == 3:
                wy, wx = wy[..., np.newaxis], wx[..., np.newaxis]
            out = (image[np.ix_(tly, tlx)] * (1-wx) * (1-wy) + image[np.ix_(tly, brx)] * wx * (1-wy)
                 + image[np.ix_(bry, tlx)] * (1-wx) *    wy  + image[np.ix_(bry, brx)] * wx *    wy)
        else:
            print('Error: Bad -method argument {}. Must be one of \'bilinear\', \'bicubic\', and \'nearest\''.format(method))
        if   image.dtype == 'uint8' : out = np.uint8(out + 0.5)
        elif image.dtype == 'uint16': out = np.uint16(out + 0.5)
    return out
    
# import png
# def saveUint16(path,z):
#     # Use pypng to write zgray as a grayscale PNG.
#     with open(path, 'wb') as f:
#         writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
#         zgray2list = z.tolist()
#         writer.write(f, zgray2list)

# def depthToint16(dMap, minVal=0, maxVal=10):
#     #Maximum and minimum distance of interception 
#     dMap[dMap>maxVal] = maxVal
#     # print(np.max(dMap),np.min(dMap))
#     dMap = ((dMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
#     return dMap

# def normalizationDepth(depthfile, savepath):
#     correctDepth = readDepth(depthfile)
#     depth = depthToint16(correctDepth, 0, 10)
#     saveUint16(depth,savepath)

