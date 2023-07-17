import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import minmax_scale
import pywt
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="STSong"

fs = 200 # 采样率
ftype = 'lowpass' # 滤波器类型
fc = 50 # 截止频率
order = 5 # 阶数
wtype = 'db4' # 小波基函数
level = 4 # 分解层数
trule = 'soft' # 阈值规则

for j in range(1,10): # 遍历附件1~9的文件夹
    if j == 8:
        n = 30
    else:
        n = 20
    for i in range(1, n + 1):  # 遍历每个文件夹中的文本文档
        filename = str(i) + '.txt' # 文本文档的名称
        x = np.loadtxt('附件' + str(j) + '/' + filename) # 读数据，加上文件夹路径
        b, a = butter(order, fc/(fs/2), ftype) # 设计低通滤波器
        x1 = filtfilt(b, a, x) # 对信号进行低通滤波
        x2 = minmax_scale(x1, feature_range=(-1,1)) # 归一化
        coeffs = pywt.wavedec(x2, wtype, level=level) # 小波分解
        new_coeffs = [] # 初始化去噪后的小波系数
        for c in coeffs: # 遍历小波系数
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 # 用中值绝对偏差法估计噪声方差
            thresh = sigma * np.sqrt(2 * np.log(len(x))) # 计算阈值
            new_c = pywt.threshold(c, thresh, trule) # 对小波系数进行阈值处理
            new_coeffs.append(new_c) # 将去噪后的小波系数添加到列表
        x3 = pywt.waverec(new_coeffs, wtype) # 小波重构
        np.savetxt('new_' + str(j) + '_' + filename, x3) # 预处理后的保存，注意加上附件编号

#可视化
for j in range(1,4): # 遍历附件1~3的文件夹
    for i in range(1,6): # 遍历每个文件夹中的前五个文本文档
        filename = str(i) + '.txt' # 文本文档的名称
        x = np.loadtxt('附件' + str(j) + '/' + filename) # 读数据，注意加上文件夹路径
        x3 = np.loadtxt('new_' + str(j) + '_' + filename) # 读预处理后的数据，注意加上附件编号和文件夹路径
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(x, color='blue') # 原始信号
        plt.title('原信号')
        plt.xlabel('采样点')
        plt.ylabel('振幅')
        plt.subplot(212)
        plt.plot(x3, color='red') # 预处理后的信号
        plt.title('预处理后信号')
        plt.xlabel('采样点')
        plt.ylabel('归一化振幅')
        plt.tight_layout() # 调整子图间距
        plt.show()