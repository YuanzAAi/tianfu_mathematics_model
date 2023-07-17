import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,accuracy_score  # 导入评价指标
from GRNNandPSO import distance_mat, Gauss, sum_layer, output_layer

df_features = pd.read_csv('features_final.csv')  # 特征文件

# 读取MFCC样本熵文件，并转换成dataframe
sam = np.loadtxt('MFCC样本熵.txt')
sam = np.array(sam)
sam = sam.reshape(190, 3)
sam = pd.DataFrame(sam)
sam[np.isinf(sam)] = np.nan
sam = sam.fillna(method='bfill')
sam = sam.iloc[:170] # 选择前170行
df_features = df_features.iloc[:170]

df_features = pd.concat([df_features, sam], axis=1)

# 定义不同的特征组合
features_1 = ['mean', 'std', 'max_value', 'min_value', 'kurt', 'skewness']  # 基本统计量
features_2 = ['se_if_1', 'se_if_2', 'se_if_3']  # 瞬时频率向量的样本熵
features_3 = ['se_ae_1', 'se_ae_2', 'se_ae_3']  # 瞬时能量向量的样本熵
features_4 = [0, 1, 2]  # MFCC样本熵

# 定义训练和测试的次数
n_trials = 10

# 创建一个空的列表，用来存放每次训练和测试的评价指标结果
results_1 = []
results_2 = []
results_3 = []
results_4 = []

# 训练和测试模型，返回评价指标
def train_and_test(features):
    X = df_features[features].values  # 提取特征矩阵
    X_train, X_test, y_train, y_test = train_test_split(X, np.array([1] * 140 + [0] * 30).reshape(-1, 1), test_size=0.2,
                                                        random_state=i)  # 构造标签，1表示天然地震，0表示非天然地震

    # 使用GRNN模型进行回归分析
    sigma = 0.5 # 设置Gauss函数的标准差
    Euclidean_D = distance_mat(X_train,X_test) # 计算距离矩阵
    Gauss_mat = Gauss(Euclidean_D,sigma) # 计算Gauss矩阵
    sum_mat = sum_layer(Gauss_mat,y_train) # 计算求和层矩阵
    y_pred = output_layer(sum_mat) # 计算输出层矩阵
    y_pred_binary = (y_pred >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_binary)  # 正确率
    mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
    mape = mean_absolute_percentage_error(y_test, y_pred)  # 平均绝对百分比误差
    r2 = r2_score(y_test, y_pred)  # 判定系数
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 均方根误差
    mse = mean_squared_error(y_test, y_pred)  # 均方误差

    return acc, mae, mape, r2, rmse, mse


# 对不同的特征组合进行多次训练和测试，将评价指标结果添加到列表中
for i in range(n_trials):
    acc_1, mae_1, mape_1, r2_1, rmse_1, mse_1 = train_and_test(features_1)
    results_1.append([acc_1, mae_1, mape_1, r2_1, rmse_1, mse_1])
    acc_2, mae_2, mape_2, r2_2, rmse_2, mse_2 = train_and_test(features_2)
    results_2.append([acc_2, mae_2, mape_2, r2_2, rmse_2, mse_2])
    acc_3, mae_3, mape_3, r2_3, rmse_3, mse_3 = train_and_test(features_3)
    results_3.append([acc_3, mae_3, mape_3, r2_3, rmse_3, mse_3])
    acc_4, mae_4, mape_4, r2_4, rmse_4, mse_4 = train_and_test(features_4)
    results_4.append([acc_4, mae_4, mape_4, r2_4, rmse_4, mse_4])


# 计算指标的均值和标准差
def get_mean_std(values):
    mean = np.mean(values)  # 计算均值
    std = np.std(values)  # 计算标准差
    return '{:.2f} ± {:.2f}'.format(mean, std)  # 返回字符串形式


# 创建一个空的dataframe，用来存放评价指标结果
df_results = pd.DataFrame(
    columns=['特征类型', '正确率', '平均绝对误差(MAE)', '平均绝对百分比误差(MAPE)', '判定系数(R2)',
             '均方根误差(RMSE)',
             '均方误差(MSE)'])

# 将不同特征组合的评价指标结果添加到dataframe中
df_results.loc[0] = ['基本统计量', get_mean_std([r[0] for r in results_1]), get_mean_std([r[1] for r in results_1]), get_mean_std([r[2] for r in results_1]), get_mean_std([r[3] for r in results_1]),
                     get_mean_std([r[4] for r in results_1]), get_mean_std([r[5] for r in results_1])]
df_results.loc[1] = ['瞬时频率向量的样本熵', get_mean_std([r[0] for r in results_2]), get_mean_std([r[1] for r in results_2]), get_mean_std([r[2] for r in results_2]), get_mean_std([r[3] for r in results_2]),
                     get_mean_std([r[4] for r in results_2]), get_mean_std([r[5] for r in results_2])]
df_results.loc[2] = ['瞬时能量向量的样本熵', get_mean_std([r[0] for r in results_3]), get_mean_std([r[1] for r in results_3]), get_mean_std([r[2] for r in results_3]), get_mean_std([r[3] for r in results_3]),
                     get_mean_std([r[4] for r in results_3]), get_mean_std([r[5] for r in results_3])]
df_results.loc[3] = ['MFCC样本熵', get_mean_std([r[0] for r in results_4]), get_mean_std([r[1] for r in results_4]), get_mean_std([r[2] for r in results_4]), get_mean_std([r[3] for r in results_4]),
                     get_mean_std([r[4] for r in results_4]), get_mean_std([r[5] for r in results_4])]

# 输出dataframe到一个Excel表格，并保存
df_results.to_excel('resultsGRNN.xlsx', index=False)
print(df_results)