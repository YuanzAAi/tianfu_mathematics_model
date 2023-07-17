import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from PyEMD import EMD
from scipy.signal import hilbert
from 样本熵 import SampEn
import pandas as pd
plt.rcParams["font.family"]="STSong"

fs = 200 # 采样率
features = {} # 特征字典

plot_frequency_list = []#瞬时频率向量画图列表
plot_energy_list = []#瞬时能量向量画图列表

for j in range(1, 10):  # 遍历附件1~9中的地震事件
    feature_matrix = []  # 特征矩阵
    if j == 8:
        n = 30
    else:
        n = 20
    for i in range(1, n + 1):  # 遍历每个文件夹中的文本文档
        filename = 'new_' + str(j) + '_' + str(i) + '.txt'
        x = np.loadtxt(filename)  # 读数据

        mean = np.mean(x)  # 计算均值
        std = np.std(x)  # 计算标准差
        max_value = np.max(x)  # 计算最大值
        min_value = np.min(x)  # 计算最小值
        kurt = kurtosis(x)  # 计算峰度
        skewness = skew(x)  # 计算斜度

        emd = EMD()  #实例化EMD对象
        imfs = emd.emd(x)  #EMD分解，得到IMF分量

        instantaneous_frequency_list = []#瞬时频率向量列表

        instantaneous_energy_list = []#瞬时能量向量列表

        for imf in imfs:

            analytic_signal_imf = hilbert(imf)   # 对每个IMF分量进行希尔伯特变换

            amplitude_envelope_imf = np.abs(analytic_signal_imf) # 振幅包络

            instantaneous_phase_imf = np.unwrap(np.angle(analytic_signal_imf)) # 瞬时相位

            instantaneous_frequency_imf = (np.diff(instantaneous_phase_imf) / (2.0 * np.pi) * fs)[:-1] # 计算瞬时频率

            instantaneous_energy_imf = amplitude_envelope_imf ** 2# 计算瞬时能量(不确定)

            # 将瞬时频率向量和瞬时能量向量添加到列表
            instantaneous_frequency_list.append(instantaneous_frequency_imf)
            plot_frequency_list.append(instantaneous_frequency_imf)
            instantaneous_energy_list.append(instantaneous_energy_imf)
            plot_energy_list.append(instantaneous_energy_imf)

        se_if_list = []# 瞬时频率向量的样本熵列表

        se_ae_list = []# 瞬时能量向量的样本熵列表

        for k in range(3):#遍历瞬时频率向量列表

            se_if = SampEn(instantaneous_frequency_list[k],2,0.2*np.std(instantaneous_frequency_list[k]))#每个瞬时频率向量的样本熵

            se_ae = SampEn(instantaneous_energy_list[k],2,0.2*np.std(instantaneous_energy_list[k])) # 每个瞬时能量向量的样本熵


            se_if_list.append(se_if)


            se_ae_list.append(se_ae)

        feature_vector = [mean, std, max_value, min_value, kurt, skewness] + se_if_list + se_ae_list # 将每个样本的特征向量组合


        feature_matrix.append(feature_vector) # 将每个样本的特征向量添加到特征矩阵中
    if len(feature_matrix) < 30:  # 用None来填充剩余的行数，使得每个特征矩阵都是30行，12列
        feature_matrix.extend([None] * (30 - len(feature_matrix)))

    features['附件' + str(j)] = feature_matrix  # 将每个地震事件的特征矩阵添加到特征字典中，以附件编号作为键

df_features = pd.DataFrame(features)  # 将特征字典转换成一个dataframe，每一列为一个地震事件的特征矩阵，共9列，30行
df_features.to_csv('features.csv', index=False)  # 将dataframe保存为一个csv文件，方便后续导入使用

# 可视化处理(以附件1的样本1为例子)
#EMD波形图
x3 = np.loadtxt('new_1_1.txt') #处理后的信号

emd = EMD()# 实例化EMD对象
imfs = emd.emd(x3)# 对预处理后的信号进行EMD分解，得到IMF分量

for i in range(len(imfs)):
    plt.subplot(len(imfs), 1, i + 1)
    plt.plot(imfs[i], color='red')
    plt.title('IMF{}'.format(i + 1))
plt.xlabel('采样点')
plt.show()

#瞬时频率向量和瞬时能量向量的样本熵波形图
se_if_list = []
se_ae_list = []
for k in range(3): # 遍历瞬时频率向量列表（只取到imf3）
    se_if = []
    se_ae = []
    for n in range(4000, 8001): # 遍历不同的样本长度
        se_if.append(SampEn(plot_frequency_list[k][:n],2,0.2*np.std(plot_frequency_list[k][:n]))) # 每个瞬时频率向量的样本熵
        se_ae.append(SampEn(plot_energy_list[k][:n],2,0.2*np.std(plot_energy_list[k][:n]))) # 每个瞬时能量向量的样本熵
    se_if_list.append(se_if) # 将每个瞬时频率向量的样本熵添加到列表
    se_ae_list.append(se_ae) # 将每个瞬时能量向量的样本熵添加到列表

plt.figure(figsize=(10, 6))
for i in range(3):  # 只画出前三个IMF分量对应的图
    plt.subplot(3, 2, i * 2 + 1)
    plt.plot(range(4000, 8001), se_if_list[i], color='blue')# 样本长度-样本熵图
    plt.title('IMF{}的瞬时频率样本熵'.format(i + 1))
    plt.xlabel('样本点')
    plt.ylabel('样本熵')

    plt.subplot(3, 2, i * 2 + 2)
    plt.plot(range(4000, 8001), se_ae_list[i], color='red')
    plt.title('IMF{}的瞬时能量样本熵'.format(i + 1))
    plt.xlabel('样本点')
    plt.ylabel('样本熵')

plt.tight_layout()
plt.show()