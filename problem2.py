import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="STSong"
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
nums = [4.2, 5.0, 6.0, 6.4, 7.0, 7.4, 8.0]
counts = [20, 20, 20, 20, 20, 20, 20]
y = []
# 建立模型，需要换模型预测时，就改这里，然后下面导出结果的文件也记得改
k = 300 #对每个特征，训练和预测k次
#MLPmodel = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=5000, random_state=42)
model = LinearSVR(C=0.5, epsilon=0.05, loss='squared_epsilon_insensitive', max_iter=100000, tol=1e-3, dual=False)
#model = SVR(kernel='rbf', C=1.2, epsilon=0.1)
# 创建Random Forest回归器，并设置参数
#model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None,min_samples_split=2, min_samples_leaf=1, max_features='auto')



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


#MFCC MSE
pre_y_df = pd.DataFrame(index=range(1000), columns=range(8))
MSE_list = []
for i in range(k):
    # 训练模型,x为训练集， y为标签
    x_train, x_test, y_train, y_test = train_test_split(sam.drop(range(140, 190), axis=0), y, test_size=0.2)
    model.fit(x_train, y_train)
    pre_y = model.predict(x_test)
    MSE = mean_squared_error(y_test, pre_y)
    MSE_list.append(MSE)
    for j in range(8):
        if j != 7:
            x_test =sam.iloc[ j  * 20:(j+1) * 20]
            # 预测,test为测试集,不断更改
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i,j] = np.mean(pre_y)
        else:
            x_test = sam.iloc[170:,]
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i, j] = np.mean(pre_y)
average1_df = pre_y_df.mean(axis=0)
average1_df = average1_df.to_frame().T
Result_MSE = np.mean(MSE_list)
average1_df[average1_df.shape[1]] = Result_MSE
print(average1_df)

#一般特征MSE
pre_y_df = pd.DataFrame(index=range(1000), columns=range(8))
MSE_list = []
feature11 = feature.iloc[:,:6]
for i in range(k):
    # 训练模型,x为训练集， y为标签
    x_train, x_test, y_train, y_test = train_test_split(feature11.drop(range(140, 190), axis=0), y, test_size=0.2)
    model.fit(x_train, y_train)
    pre_y = model.predict(x_test)
    MSE = mean_squared_error(y_test, pre_y)
    MSE_list.append(MSE)
    for j in range(8):
        if j != 7:
            x_test =feature11.iloc[ j  * 20:(j+1) * 20]
            # 预测,test为测试集,不断更改
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i,j] = np.mean(pre_y)
        else:
            x_test = feature11.iloc[170:,]
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i, j] = np.mean(pre_y)
average2_df = pre_y_df.mean(axis=0)
average2_df = average2_df.to_frame().T
Result_MSE = np.mean(MSE_list)
average2_df[average2_df.shape[1]] = Result_MSE
print(average2_df)


#HHT瞬时频率样本熵 MSE
pre_y_df = pd.DataFrame(index=range(1000), columns=range(8))
MSE_list = []
feature12 = feature.iloc[:,6:9]
for i in range(k):
    # 训练模型,x为训练集， y为标签
    x_train, x_test, y_train, y_test = train_test_split(feature12.drop(range(140, 190), axis=0), y, test_size=0.2)
    model.fit(x_train, y_train)
    pre_y = model.predict(x_test)
    MSE = mean_squared_error(y_test, pre_y)
    MSE_list.append(MSE)
    for j in range(8):
        if j != 7:
            x_test =feature12.iloc[ j  * 20:(j+1) * 20]
            # 预测,test为测试集,不断更改
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i,j] = np.mean(pre_y)
        else:
            x_test = feature12.iloc[170:,]
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i, j] = np.mean(pre_y)
average3_df = pre_y_df.mean(axis=0)
average3_df = average3_df.to_frame().T
Result_MSE = np.mean(MSE_list)
average3_df[average3_df.shape[1]] = Result_MSE
print(average3_df)

#HHT瞬时能量样本熵 MSE
pre_y_df = pd.DataFrame(index=range(1000), columns=range(8))
MSE_list = []
feature12 = feature.iloc[:,9:]
for i in range(k):
    # 训练模型,x为训练集， y为标签
    x_train, x_test, y_train, y_test = train_test_split(feature12.drop(range(140, 190), axis=0), y, test_size=0.2)
    model.fit(x_train, y_train)
    pre_y = model.predict(x_test)
    MSE = mean_squared_error(y_test, pre_y)
    MSE_list.append(MSE)
    for j in range(8):
        if j != 7:
            x_test =feature12.iloc[ j  * 20:(j+1) * 20]
            # 预测,test为测试集,不断更改
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i,j] = np.mean(pre_y)
        else:
            x_test = feature12.iloc[170:,]
            pre_y = model.predict(x_test)
            pre_y_df.iloc[i, j] = np.mean(pre_y)
average4_df = pre_y_df.mean(axis=0)
average4_df = average4_df.to_frame().T
Result_MSE = np.mean(MSE_list)
average4_df[average4_df.shape[1]] = Result_MSE
print(average4_df)

# 创建结果数据框
result_p2_MLP_df = pd.DataFrame(index=range(6), columns=range(9))

# 列命名
column_names = ['地震事件1', '地震事件2', '地震事件3', '地震事件4', '地震事件5', '地震事件6', '地震事件7', '地震事件9', 'MSE']
result_p2_MLP_df.columns = column_names

# 行命名
row_names = ['一般特征预测震级', 'MFCC样本熵预测震级', 'HHT瞬时频率样本熵预测震级', 'HHT瞬时能量样本熵预测震级', '实际震级', '平均震级差']
result_p2_MLP_df.index = row_names

# 填充前四行数据
result_p2_MLP_df.iloc[:4, :] = [average2_df.values.flatten(), average1_df.values.flatten(), average3_df.values.flatten(), average4_df.values.flatten()]

# 填充第五行数据
result_p2_MLP_df.iloc[4, :] = [4.2, 5.0, 6.0, 6.4, 7.0, 7.4, 8.0, np.nan, np.nan]

# 将空值替换为NaN
result_p2_MLP_df = result_p2_MLP_df.replace({None: np.nan})

# 计算平均震级差并填充第六行数据
diff_avg = result_p2_MLP_df.iloc[4, :-1].values - result_p2_MLP_df.iloc[:4, :-1].values
avg_diff = np.nanmean(diff_avg, axis=0)
result_p2_MLP_df.iloc[5, :-1] = avg_diff

# 打印结果
print(result_p2_MLP_df)
result_p2_MLP_df.to_excel('result_p2_LS-SVM.xlsx', index=True)