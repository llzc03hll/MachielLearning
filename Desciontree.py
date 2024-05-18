import numpy as np
import pandas as pd

class Desciontree(object):
    def __init__(self) :
            pass

    def _get_feature_values(data):
        n_features = data.shape[1] #得到特征的维度数量
        feature_values = {}
        for i in range(n_features):
            x_feature = list(set(np.sort(data[:,i]))) #遍历每一列特征维度并进行排序以及去重处理
            tmp_values = [x_feature[0]]  # 左边插入最小值
            for j in range(1,len(x_feature)):
                tmp_values.append(round((x_feature[j - 1] + x_feature[j]) / 2, 4))#计算得到两两相邻的特征值之间的平均值作为离散特征
                tmp_values.append(x_feature[-1])  # 右边插入最大值
                feature_values[i] = tmp_values
        return feature_values

Desciontree_= Desciontree()
Desciontree_ = np.array(pd.read_excel('new.xls'))
Desciontree_ = Desciontree._get_feature_values(Desciontree_)
print(Desciontree_)