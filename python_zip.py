import sys, os, zipfile
from tqdm import tqdm
sli = 10
# 获取源路径和目标路径
data = []
data.append([r'/tt/nas/optical_flow_output/optical_flow_2k_output/Avatar1/gma_left_right_mergedepth_mv0',r'/tt/nas/optical_flow_zip/2k/gma_left_right_mergedepth_mv0'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_2k_output/Avatar1/gma_left_right_mergedepth_mv1',r'/tt/nas/optical_flow_zip/2k/gma_left_right_mergedepth_mv1'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_2k_output/Avatar1/gma_right_left_mergedepth_mv0',r'/tt/nas/optical_flow_zip/2k/gma_right_left_mergedepth_mv0'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_2k_output/Avatar1/gma_right_left_mergedepth_mv1',r'/tt/nas/optical_flow_zip/2k/gma_right_left_mergedepth_mv1'])

data.append([r'/tt/nas/optical_flow_output/optical_flow_1k_output/Avatar1/gma_left_right_mergedepth_mv0',r'/tt/nas/optical_flow_zip/1k/gma_left_right_mergedepth_mv0'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_1k_output/Avatar1/gma_left_right_mergedepth_mv1',r'/tt/nas/optical_flow_zip/1k/gma_left_right_mergedepth_mv1'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_1k_output/Avatar1/gma_right_left_mergedepth_mv0',r'/tt/nas/optical_flow_zip/1k/gma_right_left_mergedepth_mv0'])
data.append([r'/tt/nas/optical_flow_output/optical_flow_1k_output/Avatar1/gma_right_left_mergedepth_mv1',r'/tt/nas/optical_flow_zip/1k/gma_right_left_mergedepth_mv1'])
for source_dir,dest_dir in data:
    password = None
    print("源目录:", source_dir)
    print("解压到:", dest_dir)
    print("解压密码:", password)

    # 判断源路径是否合法
    if not os.path.exists(source_dir):
        print("压缩文件或压缩文件所在路径不存在！")
        exit()
    if not os.path.isdir(source_dir) and not zipfile.is_zipfile(source_dir):
        print("指定的源文件不是一个合法的.zip文件！")
        exit()

    # 如果解压到的路径不存在，则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    zfile = None
    file_lst = sorted(os.listdir(source_dir))
    n = len(file_lst)
    for i in tqdm(range(n)):
            if i % round(n/sli) == 0:
                if zfile:
                    zfile.close()
                fname = file_lst[i][-10:-4]
                fname_ = str(int(fname) + round(n/sli)-1) if i != (round(n/sli))*(sli-1) else file_lst[n-1][-10:-4]
                zfile=zipfile.ZipFile(os.path.join(dest_dir,'result_{}_to_{}.zip'.format(fname,fname_)),"w")
            fil = os.path.join(source_dir,file_lst[i])
            zfile.write(fil)
TC@2020g 
