import numpy as np

def SampEn(U, m, r):
    """
    用于量化时间序列的可预测性
    :param U: 时间序列
    :param m: 模板向量维数
    :param r: 距离容忍度，一般取0.1~0.25倍的时间序列标准差，也可以理解为相似度的度量阈值
    :return: 返回一个-np.log(A/B)，该值越小预测难度越小
    """

    def _maxdist(x_i, x_j):
        """
         Chebyshev distance
        :param x_i:
        :param x_j:
        :return:
        """
        return np.max(np.abs(x_i - x_j))

    def _phi(m):
        x = np.array([U[j:j + m] for j in range(N - m + 1)])
        C = np.zeros(len(x)) # 计数数组
        for i in range(len(x)): # 遍历每个模板向量
            dist = np.max(np.abs(x - x[i]), axis=1) # 计算每个模板向量与其他模板向量的切比雪夫距离，使用numpy的向量化操作
            dist = np.sort(dist) # 对距离进行快速排序，使用numpy的排序函数
            index = np.searchsorted(dist, r, side='right') # 使用二分搜索找到第一个大于r的距离的索引
            C[i] = index - 1 # 计算小于等于r的距离的个数，要减去自身
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))
