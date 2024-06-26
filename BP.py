import math
import random

'''
BP神经网络的算法实现
'''

# 用于设置权重矩阵的大小并给定初始权重
def weight_matrix(row,col,weight = 0.0):
    weightMat = []
    for _ in range(row):
        weightMat.append([weight] * col) #weightMat就成为了一个row x col的二维列表
    return weightMat

# 用于给权重矩阵内的每元素生成一个初始随机权重
def random_weight(parm_1, parm_2):
    return (parm_1 - 1) * random.random() + parm_2

# Sigmoid激活函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Sigmoid激活函数的导函数
def sigmoid_derivate(x):
    return x * (1 - x)

# 定义BP神经网络类
class BPNeualNetwork:
    def __init__(self):
        # 定义输入层、隐藏层、输出层，所有层的神经元个数都初始化为0
        self.input_num, self.hidden_num, self.output_num = 0, 0, 0

        # 定义输入层、隐藏层、输出层的值矩阵，并在setup函数中初始化
        self.input_values, self.hidden_values, self.output_values = [], [], []

        # 定义输入-隐藏层、隐藏-输出层权重矩阵，并在setup函数中设置大小并初始化
        self.input_hidden_weights, self.hidden_output_weights = [], []

    # 神经网络的初始化函数
    # 四个参数分别代表：对象自身、输入层神经元个数、隐藏层神经元个数、输出层神经元个数
    def setup(self, input_num, hidden_num, output_num):
        # 设置输入层、隐藏层、输出层的神经元个数，其中输入层包含偏置项因此数量+1
        self.input_num, self.hidden_num, self.output_num = input_num + 1, hidden_num, output_num

        # 初始化输入层、隐藏层、输出层的值矩阵，均初始化为1
        self.input_values = [1.0] * self.input_num
        self.hidden_values = [1.0] * self.hidden_num
        self.output_values = [1.0] * self.output_num

        # 设置输入-隐藏层、隐藏-输出层权重矩阵的大小
        self.input_hidden_weights = weight_matrix(self.input_num, self.hidden_num)
        self.hidden_output_weights = weight_matrix(self.hidden_num, self.output_num)

        # 初始化输入-隐藏层、隐藏-输出层的权重矩阵
        for i in range(self.input_num):
            for h in range(self.hidden_num):
                self.input_hidden_weights[i][h] = random_weight(-0.2, 0.2)
        for h in range(self.hidden_num):
            for o in range(self.output_num):
                self.hidden_output_weights[h][0] = random_weight(-0.2, 0.2)
    #神经网络的前向预测
    def predict(self, data):
        # 将数据放入输入层，-1是由于输入层中的偏置项不需要接收数据
        for i in range(self.input_num - 1):
            self.input_values[i] = data[i] 
        # 隐藏层计算
        for h in range(self.hidden_num):
            # 激活函数的参数
            total = 0.0
            # 激活函数的参数值由输入层权重和输入层的值确定
            for i in range(self.input_num):
                total += self.input_values[i] * self.input_hidden_weights[i][h]
            # 将经过激活函数处理的输入层的值赋给隐藏层
            self.hidden_values[h] = sigmoid(total - 0)

        # 输出层计算
        for o in range(self.output_num):
            total = 0.0
            for h in range(self.hidden_num):
                total += self.hidden_values[h] * self.hidden_output_weights[h][o]
            self.output_values[o] = sigmoid(total - 0)
        return self.output_values[:]

    # 神经网络的反向传播
    # 四个参数分别代表：对象自身、单个数据、数据对应的标签、学习率(步长)
    def back_propagate(self, data, label, learn):
        # 反向传播前先进行前向预测
        self.predict(data)
        # 计算输出层的误差
        output_datas = [0.0] * self.output_num
        for o in range(self.output_num):
            error = label[o] - self.output_values[o]
            output_datas[o] = sigmoid_derivate(self.output_values[o]) * error
        # 计算隐藏层的误差
        hidden_datas = [0.0] * self.hidden_num
        for h in range(self.hidden_num):
            error = 0.0
            for o in range(self.output_num):
                error += output_datas[o] * self.hidden_output_weights[h][o]
            hidden_datas[h] = sigmoid_derivate(self.hidden_values[h]) * error
        # 更新隐藏-输出层权重
        for h in range(self.hidden_num):
            for o in range(self.output_num):
                self.hidden_output_weights[h][o] += learn * output_datas[o] * self.hidden_values[h]
        # 更新输入-隐藏层权重
        for i in range(self.input_num):
            for h in range(self.hidden_num):
                self.input_hidden_weights[i][h] += learn * hidden_datas[h] * self.input_values[i]
        # 计算样本的均方误差
        error = 0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_values[o]) ** 2
        return error
    