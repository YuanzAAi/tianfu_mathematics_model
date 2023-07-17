import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="STSong"
nums = [4.2, 5.0, 6.0, 6.4, 7.0, 7.4, 8.0]
counts = [20, 20, 20, 20, 20, 20, 20]
y = []

for i in range(len(nums)):
    for j in range(counts[i]):
        y.append(nums[i])

sam = np.loadtxt('MFCC样本熵.txt')
sam = np.array(sam)
sam = sam.reshape(190,3)
sam = pd.DataFrame(sam)
sam[np.isinf(sam)] = np.nan
sam = sam.fillna(method= 'bfill')
feature = pd.read_csv('features_final.csv')
feature = pd.DataFrame(feature)
x_train = sam.drop(range(140, 190), axis=0)
x_test = sam.iloc[170:,]
y_train = y[:140]


fig0, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=100)
x_data = [x_train.iloc[:,0], x_train.iloc[:,1], x_train.iloc[:,2]]
colors = ['red', 'green', 'blue']
y_labels = ["静态系数", "一阶差分系数", "二阶差分系数"]
for i in range(3):
    axs[i].scatter(x_data[i], y_train, c=colors[i])
    axs[i].legend(loc='best')
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].set_xlabel("样本熵", fontdict={'size': 12})
    axs[i].set_ylabel("震级", fontdict={'size': 12}, rotation=0)
    axs[i].set_title("MFCC{}".format(y_labels[i]), fontdict={'size': 20})
fig0.autofmt_xdate()
plt.show()

#可视化，看一般特征和震级的关系
feature1 = feature.iloc[:,:6]
x_train = feature1.drop(range(140, 190), axis=0)
y_train = y[:140]
fig1, axs = plt.subplots(nrows=1, ncols=6, figsize=(20, 6), dpi=100)
x_data = [x_train.iloc[:,0], x_train.iloc[:,1], x_train.iloc[:,2],x_train.iloc[:,3],x_train.iloc[:,4],x_train.iloc[:,5]]
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
y_labels = ["均值", "标准差", "最大值",'最小值','峰度','斜度']
for i in range(6):
    axs[i].scatter(x_data[i], y_train, c=colors[i])
    axs[i].legend(loc='best')
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].set_xlabel("指标", fontdict={'size': 12})
    axs[i].set_title("{}".format(y_labels[i]), fontdict={'size': 12})
    if i == 0:
        axs[i].set_ylabel("震级", fontdict={'size': 12}, rotation=0)
fig1.autofmt_xdate()
plt.show()

#可视化，看HTT瞬时频率样本熵和震级的关系
feature2 = feature.iloc[:, 6:9]
x_train = feature2.drop(range(140, 190), axis=0)
y_train = y[:140]
fig2, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=100)
x_data = [x_train.iloc[:,0], x_train.iloc[:,1], x_train.iloc[:,2]]
colors = ['red', 'green', 'blue']
y_labels = ["HTT瞬时频率样本熵(IMF1)", "HTT瞬时频率样本熵(IMF2)", "HTT瞬时频率样本熵(IMF3)"]
for i in range(3):
    axs[i].scatter(x_data[i], y_train, c=colors[i])
    axs[i].legend(loc='best')
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].set_xlabel("样本熵", fontdict={'size': 12})
    axs[i].set_ylabel("震级", fontdict={'size': 12}, rotation=0)
    axs[i].set_title("{}".format(y_labels[i]), fontdict={'size': 12})
fig2.autofmt_xdate()
plt.show()

#可视化，看HTT瞬时能量样本熵和震级的关系
feature2 = feature.iloc[:,9:]
x_train = feature2.drop(range(140, 190), axis=0)
y_train = y[:140]
fig3, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=100)
x_data = [x_train.iloc[:,0], x_train.iloc[:,1], x_train.iloc[:,2]]
colors = ['red', 'green', 'blue']
y_labels = ["HTT瞬时能量样本熵(IMF1)", "HTT瞬时能量样本熵(IMF2)", "HTT瞬时能量样本熵(IMF3)"]
for i in range(3):
    axs[i].scatter(x_data[i], y_train, c=colors[i])
    axs[i].legend(loc='best')
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].set_xlabel("样本熵", fontdict={'size': 12})
    axs[i].set_ylabel("震级", fontdict={'size': 12}, rotation=0)
    axs[i].set_title("{}".format(y_labels[i]), fontdict={'size': 12})
fig3.autofmt_xdate()
plt.show()
