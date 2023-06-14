import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = 'C:/Users/86178/Desktop/论文/结果.xlsx'
data_frame = pd.read_excel(file,sheet_name = 1)
data_values = data_frame.values[1:,1:]

import random
import matplotlib
def draw_1(index,x_ticks,colors,title,index1,x_ticks1,colors1,title1):
    matplotlib.rc('font', family='SimHei', weight='bold')
    city_name = ['Caltech101','Food101','EuroSAT','FGVCAircraft','SUN397']
    city_name.reverse()
    data = []
    for i,j in enumerate(data_values[-1] - data_values[2]):
        if i % 3 == index and i < 15:
            data.append(j)
    draw_2 = { }
    for i in range(len(city_name)):
        draw_2[city_name[i]] = data[i]

    sorted_d = sorted(draw_2.items(), key=lambda x: x[1])
    x =[]
    y = []
    for i in sorted_d:
        x.append(i[0])
        y.append(i[1])
    colors.reverse()

    data = []
    for i,j in enumerate(data_values[-1] - data_values[2]):
        if i % 3 == index1 and i < 15:
            data.append(j)
    draw_2 = { }
    for i in range(len(city_name)):
        draw_2[city_name[i]] = data[i]

    sorted_d = sorted(draw_2.items(), key=lambda x: x[1])
    x1 =[]
    y1 = []
    for i in sorted_d:
        x1.append(i[0])
        y1.append(i[1])
    colors1.reverse()
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)

    plt.barh(range(len(sorted_d)), y, tick_label=x, color=colors)

    plt.title(title,fontsize=15)
    plt.rcParams['axes.unicode_minus'] = False
    # 不要X横坐标标签。
    plt.xlabel('Absolute improvement(%)',fontsize=15)
    plt.xticks(x_ticks,fontsize=12)

    for i,(a, b) in enumerate(zip(range(len(sorted_d)), y)):  # 柱子上的数字显示
        if i < 1:
            plt.text(b +1.8, a -0.1, '%.2f' % b, ha='center', va='bottom', fontsize=12)
        else:
            plt.text(b + 0.8, a - 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    plt.subplot(1,2,2)
    plt.barh(range(len(sorted_d)), y1, tick_label=x1, color=colors1)
    plt.title(title1,fontsize=15)
    plt.rcParams['axes.unicode_minus'] = False
    # 不要X横坐标标签。
    plt.xlabel('Absolute improvement(%)',fontsize=15)
    plt.xticks(x_ticks1,fontsize=12)
    for i,(a, b) in enumerate(zip(range(len(sorted_d)), y1)):  # 柱子上的数字显示
        if i < 4:
            plt.text(b -0.8, a -0.1, '%.2f' % b, ha='center', va='bottom', fontsize=12)
        else:
            plt.text(b - 1.6, a - 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=12)



    plt.show()

# draw_1(1, [0,4,8,12,16],['limegreen', 'limegreen', 'limegreen', 'limegreen', 'moccasin'],'(a) CSMP vs. CoOp in Unseen Classes',
#         0,[-16,-12,-8,-4,0,],['limegreen', 'moccasin', 'moccasin', 'moccasin', 'moccasin'],'(b) CSMP vs. CoOp in Base Classes')


def boxplot1():
    # 设置字体, 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决图像中的'-'负号的乱码问题
    plt.rcParams['axes.unicode_minus'] = False

    ClassA_C = [73.72,74.10]
    ClassB_C = [77.85,69.14]
    ClassC_C = [79.91,63.00]
    ClassD_C = [69.77,72.18]

    fig = plt.figure(figsize=(8, 6.1))
    ax = fig.add_subplot(facecolor='white')
    # 橙绿蓝
    color_list = ['#FF8C00', '#00FF00', '#0000FF', '#D9B3B3']
    # marker的形状列表
    marker_list = ['D', 'D', 'D', 'D']
    # marker的大小列表
    markersize_list = [8, 8, 8,8]

    x_labels = ['JSMP', 'CoCoOp', 'CoOp', 'CLIP']
    x_loc = [1, 2, 3, 4]

    boxplot_data = [ClassA_C, ClassB_C, ClassC_C,ClassD_C]


    bp = ax.boxplot(boxplot_data, positions=x_loc, widths=0.4, patch_artist=True, showfliers=True)

    for i in range(len(bp['boxes'])):
        bp['boxes'][i].set(facecolor='None', edgecolor=color_list[i])
        # 顶端, 末端两条线; 顶端:0, 2, 4; 末端:1, 3, 5
        bp['caps'][2 * i].set(color=color_list[i])
        bp['caps'][2 * i + 1].set(color=color_list[i])
        # 中位数那条线
        bp['medians'][i].set(color=color_list[i])
        # 顶端, 末端两条须; 顶端:0, 2, 4; 末端:1, 3, 5
        bp['whiskers'][2 * i].set(color=color_list[i])
        bp['whiskers'][2 * i + 1].set(color=color_list[i])
        # 分别设置异常点的形状, 填充颜色, 轮廓颜色, 大小
        bp['fliers'][i].set_marker(marker_list[i])
        bp['fliers'][i].set_markerfacecolor(color_list[i])
        bp['fliers'][i].set_markeredgecolor(color_list[i])
        bp['fliers'][i].set_markersize(markersize_list[i])

    # ax.grid(True, ls=':', color='b', alpha=0.3)
    plt.title('Average over 5 datasets', fontweight='bold',fontsize=15)
    ax.set_xticks(x_loc)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_ylabel('Accuracy(%)', fontweight='bold',fontsize=15)
    plt.xticks(weight='bold',fontsize=12)
    plt.yticks(weight='bold',fontsize=12)
    # fig.tight_layout()

    plt.show()

# boxplot1()

def ablation_manual_prompts():
    size = 3

    x = np.arange(size)

    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.7, 3
    # 每种类型的柱状图宽度
    width = total_width / n

    # 重新设置x轴的坐标
    x = x - (total_width - width) / 2

    plt.rcParams['font.serif']=['Times New Roman']
    plt.figure(figsize=(16,18))
    # 画柱状图
    plt.subplot(2,2,1)

    list1=[86.36,85.01,85.72]
    list2=[87.07,87.08,87.10]
    list3=[86.75,86.82,86.81]
    plt.bar(x, list1, width=width, label="'a photo of a'",color='#0066cc')
    plt.bar(x + width,list2, width=width, label="'a type of {category}, a photo of a'",color='#9ACD32')
    plt.bar(x + width*2,list3, width=width, label="', a type of {category}, a photo of a'",color='#FF8C00')
    for i,list in enumerate([list1,list2,list3]):
        plt.text(width*i-0.23, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
        plt.text(width*i+0.77, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
        plt.text(width * i+1.77, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)
    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(3), ('base', 'unseen', 'average'))
    # 显示图例
    plt.ylim([80,90])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(prop={"family": "Times New Roman"})
    # plt.xlabel("(a) Ablation on Manual Prompts over 3 Datasets.",fontname="Times New Roman",fontsize=15)
    plt.title("(a) Ablation on Manual Prompts over 3 Datasets.", fontname="Times New Roman", fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)

    list1=[90.25,87.17,88.70]
    list2=[91.62,90.73,91.17]
    list3=[91.15,90.86,91.02]
    # 画柱状图
    plt.subplot(2,2,2)
    plt.bar(x, list1, width=width, label="'a photo of a'",color='#0066cc')
    plt.bar(x + width,list2, width=width, label="'a type of food, a photo of a'",color='#9ACD32')
    plt.bar(x + width*2,list3, width=width, label="', a type of food, a photo of a'",color='#FF8C00')
    for i,list in enumerate([list1,list2,list3]):
        plt.text(width*i-0.23, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
        plt.text(width*i+0.77, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
        plt.text(width * i+1.77, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)
    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(3), ('base', 'unseen', 'average'))
    # 显示图例
    plt.ylim([85,95])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(prop={"family": "Times New Roman"})
    # plt.xlabel("(b) Ablation on Manual Prompts over Food101.",fontname="Times New Roman",fontsize=15)
    plt.title("(b) Ablation on Manual Prompts over Food101.", fontname="Times New Roman", fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)

    list1=[96.68,92.56,94.72]
    list2=[97.07,94.77,95.99]
    list3=[97.06,94.06,95.63]
    # 画柱状图
    plt.subplot(2,2,3)
    plt.bar(x, list1, width=width, label="'a photo of a'",color='#0066cc')
    plt.bar(x + width,list2, width=width, label="'a type of universal goods, a photo of a'",color='#9ACD32')
    plt.bar(x + width*2,list3, width=width, label="', a type of universal goods, a photo of a'",color='#FF8C00')
    for i,list in enumerate([list1,list2,list3]):
        plt.text(width*i-0.23, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
        plt.text(width*i+0.77, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
        plt.text(width * i+1.77, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)
    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(3), ('base', 'unseen', 'average'))
    # 显示图例
    plt.ylim([90,100])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(prop={"family": "Times New Roman"})
    # plt.xlabel("(c) Ablation on Manual Prompts over Caltech101.",fontname="Times New Roman",fontsize=15)
    plt.title("(c) Ablation on Manual Prompts over Caltech101.",fontname="Times New Roman",fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)

    list1=[72.16,75.31,73.74]
    list2=[72.23,76.48,74.36]
    list3=[72.04,75.53,73.79]
    # 画柱状图
    plt.subplot(2,2,4)
    plt.bar(x, list1, width=width, label="'a photo of a'",color='#0066cc')
    plt.bar(x + width,list2, width=width, label="'a type of scene, a photo of a '",color='#9ACD32')
    plt.bar(x + width*2,list3, width=width, label="',a type of scene, a photo of a '",color='#FF8C00')
    for i,list in enumerate([list1,list2,list3]):
        plt.text(width*i-0.23, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
        plt.text(width*i+0.77, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
        plt.text(width * i+1.77, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)
    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(3), ('base', 'unseen', 'average'))
    # 显示图例
    plt.ylim([70,80])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(prop={"family": "Times New Roman"})
    # plt.xlabel("(d) Ablation on Manual Prompts over Sun397.",fontname="Times New Roman",fontsize=15)
    plt.title("(d) Ablation on Manual Prompts over Sun397.",fontname="Times New Roman",fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)
    # plt.savefig('plot123_2.png',dpi=500)
    # 显示柱状图
    plt.show()


ablation_manual_prompts()













def ablation_context_length():
    #prompt length
    size = 3

    x = np.arange(size)

    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.8, 5
    # 每种类型的柱状图宽度
    width = total_width / n

    list1=[86.35,85.32,85.85]
    list2=[86.60,86.29,86.47]
    list3=[86.44,87.13,86.79]
    list4=[86.11,86.20,86.19]
    list5=[85.87,86.08,85.92]
    # 重新设置x轴的坐标
    x = x - (total_width - width) / 2
    plt.figure(figsize=(10,8))
    plt.rcParams['font.serif']=['Times New Roman']
    # 画柱状图
    plt.bar(x, list1, width=width, label="context length = 1",color='#0066cc')
    plt.bar(x + width,list2, width=width, label="context length = 2",color='#9ACD32')
    plt.bar(x + width*2,list3, width=width, label="context length = 4",color='#FF8C00')
    plt.bar(x + width*3,list4, width=width, label="context length = 8",color='#008000')
    plt.bar(x + width*4,list5, width=width, label="context length = 16",color='#FFE4B5')
    for i,list in enumerate([list1,list2,list3,list4,list5]):
        plt.text(width*i-0.32, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
        plt.text(width*i+0.68, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
        plt.text(width * i+1.68, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)


    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(np.arange(3), ('base', 'unseen', 'average'))
    # 显示图例
    plt.ylim([80,90])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(loc='lower right',prop={"family": "Times New Roman"})
    plt.xlabel("Ablation on Context length.",fontname="Times New Roman",fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)

    # plt.savefig('plot123_2.png',dpi=500)
    # 显示柱状图
    plt.show()

def plot():
    #prompt length
    size = 3

    x = np.arange(size)

    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.8, 5
    # 每种类型的柱状图宽度
    width = total_width / n
    list1 = [85.85,86.47,86.79,86.19,85.92]
    list_sun = [73.47,73.71,74.36,73.16,73.64]
    list_food = [88.00,90.07,91.17,90.99,89.67]
    list_cal = [96.09,95.62,94.85,94.42,94.45]



    x = [1,2,4,8,16]
    plt.rcParams['font.serif']=['Times New Roman']
    # 画柱状图
    plt.plot(x,list1,marker='*',linestyle='--',label="'average'",color='#0066cc')
    plt.plot(x,list_sun,marker='.',linestyle='--',label="'Sun397'",color='#9ACD32')
    plt.plot(x,list_food,marker='.',linestyle='--',label="'Food101'",color='#FF8C00')
    plt.plot(x,list_cal,marker='.',linestyle='--',label="'Caltech101'",color='#FFE4B5')
    # plt.bar(x, list1, width=width, label="'a photo of a'",color='#0066cc')
    # plt.bar(x + width,list2, width=width, label="'a type of food, a photo of a'",color='#9ACD32')
    # plt.bar(x + width*2,list3, width=width, label="', a type of food, a photo of a'",color='#FF8C00')
    # plt.bar(x + width*3,list4, width=width, label="'a type of food, a photo of a'",color='#008000')
    # plt.bar(x + width*4,list5, width=width, label="', a type of food, a photo of a'",color='#FFE4B5')
    # for i,list in enumerate([list1,list2,list3,list4,list5]):
    #     plt.text(width*i-0.32, list[0], '%.2f' % list[0], ha='center', va='bottom', fontsize=10)
    #     plt.text(width*i+0.68, list[1], '%.2f' % list[1], ha='center', va='bottom', fontsize=10)
    #     plt.text(width * i+1.68, list[2], '%.2f' % list[2], ha='center', va='bottom', fontsize=10)


    #plt.bar(x + 2*width, c, width=width, label="c")
    plt.xticks(x, x)
    # 显示图例
    plt.ylim([0,100])
    #plt.figure(dpi=300,figsize=(24,24))
    plt.legend(loc='lower right',prop={"family": "Times New Roman"})
    plt.xlabel("(a) Ablation on Food101.",fontname="Times New Roman",fontsize=15)
    plt.ylabel("accuracy(%)",fontname="Times New Roman",fontsize=15)

    # plt.savefig('plot123_2.png',dpi=500)
    # 显示柱状图
    plt.show()


