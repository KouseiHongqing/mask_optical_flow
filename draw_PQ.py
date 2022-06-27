'''
Author: Qing Hong
Date: 2022-06-07 14:43:55
LastEditors: QingHong
LastEditTime: 2022-06-21 14:21:49
Description: file content
'''
import matplotlib.pyplot as plt
import numpy as np
def avg(a):
    return np.mean(a)
#epe full mask bounding
Farneback=[2.60,2.29,3.45]
Deepflow=[2.14,1.68,2.09]
Simpleflow=[2.33,2.24,3.01]
Sparse_to_dense_flow=[2.30,2.14,2.17]
Pca_flow=[3.27,2.03,2.12]
Rlof=[3.09,2.9,2.97]
# #acc
Farneback=[82.53,85.00,81.03]
Deepflow=[87.67,90.16,88.96]
Simpleflow=[86.42,87.06,83.49]
Sparse_to_dense_flow=[86.30,87.28,87.48]
Pca_flow=[74.15,86.75,86.75]
Rlof=[78.61,80.53,81.37]
# # #speed
Farneback=[4.96,5.19,78.51]
Deepflow=[0.45,0.45,3.79]
Simpleflow=[0.45,0.45,6.45]
Sparse_to_dense_flow=[5.71,16.93,72.53]
Pca_flow=[7.25,22.45,49.85]
Rlof=[2.66,2.66,25.81]
x = ['Fully','Mask','Bounding_box']
#print([i+1 for i in a])
# d=[Farneback,Deepflow,Simpleflow,Sparse_to_dense_flow,Pca_flow,Rlof]
# for i in d:
#     i[1],i[0] = i[0],i[1]
# print(np.shape(d[0]))
# for test in d:
#     print(test)
fig, ax = plt.subplots() # 创建图实例
ax.plot(x, Farneback, label='Farneback') 
ax.plot(x, Deepflow, label='Deepflow') 
ax.plot(x, Simpleflow, label='Simpleflow') 
ax.plot(x, Sparse_to_dense_flow, label='Sparse_to_dense_flow') 
ax.plot(x, Pca_flow, label='Pca_flow') 
ax.plot(x, Rlof, label='Rlof') 
 
 
ax.set_xlabel('Search method') #设置x轴名称 x label

ax.set_ylabel('error') #设置y轴名称 y label
ax.set_title('EPE') #设置图名为Simple Plot
ax.set_ylabel('Percentage') #设置y轴名称 y label
ax.set_title('Accuracy') #设置图名为Simple Plot
ax.set_ylabel('FPS') #设置y轴名称 y label
ax.set_title('Speed') #设置图名为Simple Plot



ax.legend() #自动检测要在图例中显示的元素，并且显示
 
plt.show() #图形可视化

# import numpy as np
# import matplotlib.pyplot as plt

# labels = ['Fully','P1_Mask','Mask']
# Farneback=[86.73,63.26,87.21]
# Deepflow=[89,85.85,85.85]
# Simpleflow=[81.22,81.3,82.56]
# Sparse_to_dense_flow=[85.95,67.76,81.79]
# Pca_flow=[72.29,52.96,83.1]
# Rlof=[71.21,45.49,71.78]
# Gma_cpu=[93.65,82.21,86.99]
# data = [Farneback,Deepflow,Simpleflow,Sparse_to_dense_flow,Pca_flow,Rlof,Gma_cpu]

# x = range(len(labels))
# width = 0.35

# # 将bottom_y元素都初始化为0
# bottom_y = np.zeros(len(labels))
# data = np.array(data)
# # 按列计算计算每组柱子的总和，为计算百分比做准备
# sums = np.sum(data, axis=0)
# for i in data:
#     # 计算每个柱子的高度，即百分比
#     y = i / sums
#     plt.bar(x, y, width, bottom=bottom_y)
#     # 计算bottom参数的位置
#     bottom_y = y + bottom_y

# plt.xticks(x, labels)
# plt.title('Accuracy')
# plt.show()



# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np


# # 构建数据
# x = np.arange(17)
# # y1 = [6, 10, 4, 8, 9]
# # y2 = [1, 3, 4, 4, 5]
# y1 = [7.16,6.1,7.66,6.99,6.19,8.26,8.09,6.65,6.46,7.55,7.78,7.14,8.15,8.95,6.92,7.06,6.12]
# y2 = [19.39,20.38,21.93,23.08,24.85,23.99,23.06,22.18,23.17,21.30,19.46,22.40,23.20,23.46,23.48,23.10,23.19]

# bar_width  = 0.35
# tick_label = [str(i+1) for i in range(17)]
# # 绘制柱状图
# plt.figure(figsize=(4, 4))

# plt.bar(x, y1, bar_width, align="center", color='r', tick_label=tick_label, label='FullySearch')
# plt.bar(x+bar_width, y2, bar_width, align="center", color='b', tick_label=tick_label, label='MaskSearch')

# plt.xlabel("scene")
# plt.ylabel("accuracy")
# # 设置x轴刻度显示位置
# plt.xticks(x+bar_width/2, tick_label)

# plt.legend(loc='upper right')
# plt.show()