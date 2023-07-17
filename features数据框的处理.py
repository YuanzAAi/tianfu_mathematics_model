# 创建一个空的数据框，用来存放新的特征矩阵
import numpy as np
import pandas as pd

df_features = pd.read_excel('features.xlsx')  # 特征文件

df_new = pd.DataFrame()

# 定义新的特征名称
new_features = ['mean', 'std', 'max_value', 'min_value', 'kurt', 'skewness', 'se_if_1', 'se_if_2', 'se_if_3', 'se_ae_1', 'se_ae_2', 'se_ae_3']

# 遍历每一个新的特征名称
for i, feature in enumerate(new_features):
    # 创建一个空的列表，用来存放提取出来的元素
    result = []
    # 遍历每一个附件，即每一个列名
    for col in df_features.columns:
        # 遍历这一列的每一行
        for row in df_features[col]:
            if isinstance(row, str):
                # 就把它转换成一个数组
                row = np.array(eval(row))
                # 然后把数组的第i个元素添加到结果列表中
                result.append(row[i])
    # 把结果列表转换成一个一维数组，并添加到新的数据框中
    df_new[feature] = np.array(result)

# 输出新的数据框
print(df_new)
df_new.to_csv('features_final.csv', index=False)  # 将dataframe保存为一个csv文件，方便后续导入使用