import pandas as pd
import numpy as np
import math

#算法实现
class KNearestNeighbor(object):
    def __init__(self) :
        pass

    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X,k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        d1 = -2 * np.dot(X, self.X_train.T)   
        d2 = np.sum(np.square(X), axis=1, keepdims=True)   
        d3 = np.sum(np.square(self.X_train), axis=1)   
        dist = np.sqrt(d1 + d2 + d3)

        # 根据K值，选择最可能属于的类别
        y_pred = np.zeros(num_test)
        for i in range(num_test):
           dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
           y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
           y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

        return y_pred



# 数据集文件列表
dataset =pd.read_excel('iri.xls')
dataset.columns = ['A', 'B', 'C', 'D', 'judgesort'] 


#数据预处理
X = dataset.iloc[0:150,0:4].values
y = dataset.iloc[0:150,4].values

#print(X)
#print(y)

X_firstspicy,y_firstspicy = X[0:50],y[0:50]
X_secendspicy,y_secendspicy = X[50:100],y[50:100]
X_thirdspicy,y_thirdspicy = X[100:150],y[100:150]

'''
    此部分利用将数据集拆分成训练集、验证集、测试集
    其中比例分别为60%,10%,10%
'''
#Training set
X_firstspicy_train = X_firstspicy[:30, : ]
y_firstspicy_train = y_firstspicy[:30]
X_secendspicy_train = X_secendspicy[:30, : ]
y_secendspicy_train = y_secendspicy[:30]
X_thirdspicy_train = X_thirdspicy[:30,  : ]
y_thirdspicy_train = y_thirdspicy[:30]

X_train = np.vstack([X_firstspicy_train,X_secendspicy_train,X_thirdspicy_train])
y_train = np.hstack([y_firstspicy_train,y_secendspicy_train,y_thirdspicy_train])

#validation set 验证集
X_firstspicy_val = X_firstspicy[30:40, : ]
y_firstspicy_val = y_firstspicy[30:40]
X_secendspicy_val = X_secendspicy[30:40, : ]
y_secendspicy_val = y_secendspicy[30:40]
X_thirdspicy_val = X_thirdspicy[30:40, : ]
y_thirdspicy_val = y_thirdspicy[30:40]

X_val = np.vstack([X_firstspicy_val,X_secendspicy_val,X_thirdspicy_val])
y_val = np.hstack([y_firstspicy_val,y_secendspicy_val,y_thirdspicy_val])

#testing set 测试集
X_firstspicy_test = X_firstspicy[40:50, : ]
y_firstspicy_test = y_firstspicy[40:50]
X_secendspicy_test = X_secendspicy[40:50, : ]
y_secendspicy_test = y_secendspicy[40:50]
X_thirdspicy_test = X_thirdspicy[40:50, : ]
y_thirdspicy_test = y_thirdspicy[40:50]

X_test = np.vstack([X_firstspicy_test,X_secendspicy_test,X_thirdspicy_test])
y_test = np.hstack([y_firstspicy_test,y_secendspicy_test,y_thirdspicy_test])


KNN = KNearestNeighbor()
KNN.train(X_train, y_train)
for k in range(1, 20):
    y_pred = KNN.predict(X_val, k)
    accuracy = np.mean(y_pred == y_val)
    print('k={} , 正确率：{}'.format(k,accuracy))

KNN = KNearestNeighbor()
KNN.train(X_train, y_train)
y_pred = KNN.predict(X_test, k=20)
accuracy = np.mean(y_pred == y_test)
print('测试集预测准确率：%f' % accuracy)