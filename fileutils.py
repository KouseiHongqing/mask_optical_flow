# -*- coding: UTF-8 -*-
#
# Author: Junhua Chen, 2021
#
import os
import re

def change_ext(filename, ext):
    name, ext0 = os.path.splitext(filename)
    return name + ext

#==============================================================================
# PREPARE FILE LIST
#==============================================================================
def list_from_first_file_name(firstfilename, n=0):
    '''
    如果文件名中不包含数字，则返回只包含该文件的list；
    如果文件名中包含数字，且scan == True，则统计以各数字片段为变化序号的文件名数量，并
        取文件名数量最多的数字片段作为文件名变化序号。
    如果文件名中包含数字，且scan == False，则选取文件名中最长的数字部分最为索引，输出n帧文件名的list；
    '''
    dirname = os.path.dirname(firstfilename)
    filename = os.path.basename(firstfilename)
    basename = os.path.splitext(filename)[0] # file name w/o ext
    r = re.compile('\\d+')
    numlist = [[x.start(),x.end()] for x in r.finditer(basename)] # 文件名里各数字list
    if len(numlist) > 0:
        numlenlist = [x[1] - x[0] for x in numlist]
        idx = numlenlist.index(max(numlenlist))
        left, right = numlist[idx][0], numlist[idx][1]
        fmt = '{}%{}{}d{}'.format(filename[:left], '0' if filename[left]=='0' else '', right-left, filename[right:])
        start = int(basename[left:right])
        if n > 0:
            outputlist = [os.path.join(dirname, fmt % (start+i)) for i in range(n)]
        else:
            outputlist = []
            while True:
                filename = os.path.join(dirname, fmt % start)
                if not os.path.isfile(filename): break;
                outputlist.append(filename)
                start += 1
    else: # 文件名里不包含数字
        outputlist = [firstfilename]
    return outputlist

def decode_file_name(path):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    basename = os.path.splitext(filename)[0] # file name w/o ext
    r = re.compile('\\d+')
    numlist = [[x.start(),x.end()] for x in r.finditer(basename)] # 文件名里各数字list
    if len(numlist) > 0:
        numlenlist = [x[1] - x[0] for x in numlist]
        idx = numlenlist.index(max(numlenlist))
        left, right = numlist[idx][0], numlist[idx][1]
        fmt = '{}%{}{}d{}'.format(filename[:left], '0' if filename[left]=='0' else '', right-left, filename[right:])
        fmt = os.path.join(dirname, fmt)
        start = int(basename[left:right])
    else: # 文件名里不包含数字
        fmt, start = path, None
    return fmt, start

def build_io_list(opt, ext=None, in_ext=None):
    '''
    指定输入方法：
    -i dir_name             指定文件夹，则该文件夹下所有图像文件，不管-n选项
    -i xxx%dxxx -istart n   指定文件名格式及第一帧序号。只处理连续序号，所以实际处理帧数可能小于`n`
    -i xxxx.xxx -n len      指定第一帧文件名及处理长度。如果len为0，则持续到所有文件。
                            对第一帧文件名作分析，提取文件名中的所有数字片段，然后统计以各数字片段为变化序号的文件名数量，
                            并取文件名数量最多的数字片段作为文件名变化序号。对这些按该序号变化的文件名排序。如果文件名中
                            没有数字，则选择所有图像文件，对文件名排序。然后抽取从当前帧开始的`len`帧（若不足`len`帧，
                            则到结尾结束）。
    -all                    处理输入文件文件夹里的所有图像文件，不管-n选项

    -o dir_name             指定文件夹，输出文件名和输入文件名一样，但统一用png作extension
    -o xxx%dxxx -ostart n   指定文件名格式及第一帧序号
    -o xxxx.xxx             指定第一帧文件名。文件名里必须包含数字，否则只处理一帧并输出。数字最长的部分被作为序列索引。
    -ref dir_name           参考图像文件夹，用于计算PSNR等目的。可选。
                            如果该文件夹下有和输入文件名相同的文件名，则该文件名就是参考文件名；
                            否则如果有相同的文件名但扩展名不同，则该文件名就是参考文件名；
                            否则如果文件名是输入文件名加'x.'（其中'.'表示任意字符，如正则表达式规定；例如'x2'），扩展名为
                            图像文件扩展名，则该文件名就是参考名。
    
    返回：
    inputlist, outputlist, reflist
    输入、输出、参考文件列表。
    如果没有opt.output属性，outputlist为 None
    如果没有opt.ref属性，reflist为None。如果存在opt.ref属性，reflist可能为[]（空列表）

    注：
    1. 选项"-n 0"和"-all"是有区别的：前者是从指定帧开始持续到结尾，而后者则简单地包括所有图像文件。
    2. `opt`中可以不包含`output`、`all`和`ref`属性。

    '''
    #======== build input list ========
    imgexts = ('.png', '.Png', '.PNG', '.jpg', '.jpeg', '.Jpg', '.Jpeg', '.JPG', '.JPEG',
                '.bmp', '.Bmp', '.BMP', '.tif', '.tiff', '.Tif', '.Tiff', '.TIF', '.TIFF', '.exr')
    videxts = ('.mp4', '.Mp4', '.MP4', '.mkv', '.Mkv', '.MKV', '.mov', '.Mov', '.MOV',
               '.avi', '.Avi', '.AVI', '.ts', '.Ts', '.TS')
    if in_ext != None:
        imgexts = in_ext

    # 获取某个目录下的所有图像文件
    def retrieve_images(apath, n=0):
        imnamelist = [x for x in os.listdir(apath) if x.endswith(imgexts) and not x.startswith('.')]
        imnamelist.sort()
        if n > 0:
            imnamelist = imnamelist[:n]
        filelist = [os.path.join(apath, x) for x in imnamelist]
        return filelist, imnamelist

    inputlist = []
    if os.path.isdir(opt.input): # all images in the folder
        inputlist, imnamelist = retrieve_images(opt.input, opt.n if hasattr(opt, 'n') else 0)
    elif opt.input != '':
        if hasattr(opt, 'all') and opt.all: # all images in the folder
            inputlist, imnamelist = retrieve_images(os.path.dirname(opt.input))
        elif opt.input.endswith(videxts) and os.path.isfile(opt.input): # a video file
            import math
            import imageio
            vid = imageio.get_reader(opt.input, 'ffmpeg')
            info = vid.get_meta_data()
            if 'nframes' in info and not math.isinf(info['nframes']):
                n = info['nframes']
            else:
                import cv2
                try:
                    cap = cv2.VideoCapture(opt.input)
                    if not cap.isOpened(): 
                        raise Exception
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                except:
                    return None, None, None
                if hasattr(opt, 'n') and opt.n > 0:
                    n = min(n, opt.n)
                inputlist = [os.path.splitext(opt.input)[0]] * n
                imnamelist = [os.path.basename(i) for i in inputlist]
        elif opt.input.find('%') >= 0: # file name with %d
            inputlist = []
            for i in range(opt.istart, opt.istart + (100000000 if opt.n == 0 else opt.n)):
                fn = opt.input % i
                if os.path.isfile(fn): inputlist.append(fn)
                else: break
            imnamelist = [os.path.basename(x) for x in inputlist]
        else: # file name
            dirname = os.path.dirname(opt.input)
            if dirname == '': dirname = '.'
            filename = os.path.basename(opt.input)
            basename, extname = os.path.splitext(filename) # file name w/o ext
            candlist = [x for x in os.listdir(dirname) if x.endswith(extname)] # 所有图像文件

            r = re.compile('\\d+')
            numlist = [[x.start(),x.end()] for x in r.finditer(basename)] # 文件名里各数字list
            if len(numlist) > 0:
                grplist = dict()
                for num in numlist:
                    r = re.compile(basename[0:num[0]] + '\\d+' + basename[num[1]:]) # 把数字换成`\d+`进行搜索
                    grplist['{}_{}'.format(num[0], num[1])] = list(filter(r.match, candlist)) # 搜索的结果作为list，放入字典
                max_key = max(grplist, key=lambda x: len(grplist[x])) # 找到文件名最多的list的key
                seqlist = sorted(grplist[max_key])
            else: # 文件名里不包含数字，则排序所有图像文件
                seqlist = sorted(candlist)

            index = seqlist.index(filename) # 当前文件在sequence里的index
            last = min(index + (len(seqlist) if opt.n == 0 else opt.n), len(seqlist))
            imnamelist = seqlist[index:last]
            inputlist = [os.path.join(dirname, x) for x in imnamelist]

    # print('\n'.join(inputlist))
    # print('\n'.join(imnamelist))

    #======== build output list ========
    if hasattr(opt, 'output') and opt.output != '' and opt.output is not None:
        # outext = ext if ext != None else '.png'
        if ext is not None:
            if ext[0] != '.':
                ext = '.' + ext
            outext = ext
        elif len(inputlist) > 0:
            outext = '.png' if inputlist[0].endswith(('.jpg', '.Jpg', '.JPG', '.bmp', '.Bmp', '.BMP')) else os.path.splitext(inputlist[0])[-1]
        else:
            outext = '.png'

        if opt.output.find('%') >= 0: # file name with %d
            outputlist = [opt.output % i for i in range(opt.ostart, opt.ostart + len(inputlist))]
        elif opt.output.endswith(imgexts): # the first output file name
            dirname = os.path.dirname(opt.output)
            filename = os.path.basename(opt.output)
            basename = os.path.splitext(filename)[0] # file name w/o ext
            r = re.compile('\\d+')
            numlist = [[x.start(),x.end()] for x in r.finditer(basename)] # 文件名里各数字list
            if len(numlist) > 0:
                numlenlist = [x[1] - x[0] for x in numlist]
                idx = numlenlist.index(max(numlenlist))
                left, right = numlist[idx][0], numlist[idx][1]
                fmt = '{}%{}{}d{}'.format(filename[:left], '0' if filename[left]=='0' else '', right-left, filename[right:])
                ostart = int(basename[left:right])
                outputlist = [os.path.join(dirname, fmt % (ostart+i)) for i in range(len(inputlist))]
            else:
                outputlist = [opt.output]
        else: # output path (directory)
            outputlist = [os.path.join(opt.output, os.path.splitext(x)[0])+outext for x in imnamelist]

        if len(outputlist) > 0 and os.path.dirname(outputlist[0]) != '' and not os.path.exists(os.path.dirname(outputlist[0])):
            os.makedirs(os.path.dirname(outputlist[0]))
    else:
        outputlist = None # [''] * len(inputlist) # changed 2021/6/22

    #======== build ref list ========
    # 查看root和exts里的元素结合后的路径是否为一个文件，若是，则返回结合后的路径；若所有的扩展名都测试后都不成功，则返回空字符串
    # root: 去除扩展名的路径
    # exts: 扩展名的集合
    def test_different_ext(root, exts):
        for ext in exts:
            if os.path.isfile(root + ext):
                return root + ext
        return ''

    reflist = None # [''] * len(inputlist) # changed 2021/7/29
    if hasattr(opt, 'ref') and opt.ref is not None:
        if os.path.isdir(opt.ref): # 指定了ref且ref目录存在
            reflist = []
            for i in range(0, len(imnamelist)):
                refile = os.path.join(opt.ref, imnamelist[i])
                if os.path.isfile(refile): # 有和输入相同的文件名
                    reflist.append(refile)
                else:
                    testname, testext = os.path.splitext(refile)
                    refname = test_different_ext(testname, imgexts)
                    if refname == '' and testname[-1].isdigit() and testname[-2].lower() == 'x':
                        refname = test_different_ext(testname[0:-2], imgexts)
                    if refname != '':
                        reflist.append(refname)
            if len(reflist) == 0: # 没有找到和输入相同（或部分相同）的文件名，则对该目录下所有图像文件排序，取前面若干文件
                reflist = sorted([os.path.join(opt.ref, x) for x in os.listdir(opt.ref) if x.endswith(imgexts)])[:len(inputlist)]
        elif not os.path.isfile(opt.ref):
            print('Warning: {} is not a file or path.'.format(opt.ref))
        elif not opt.ref.endswith(imgexts): # 是图像文件
            print('Warning: {} is not an image file.'.format(opt.ref))
        else:
            reflist = list_from_first_file_name(opt.ref, len(inputlist))

    return inputlist, outputlist, reflist
