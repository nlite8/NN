
# import numpy as np
# import math
#
#
# # 定义Sigmoid函数
# def get(x):
#     act_vec = []
#     for i in x:
#         act_vec.append(1/(1+math.exp(-i)))
#     act_vec = np.array(act_vec)
#     return act_vec
#
#
# # 训练BP神经网络
# def TrainNetwork(sample, label):
#     sample_num = len(sample)
#     sample_len = len(sample[0])
#     out_num = 2
#     hid_num = 4
#     w1 = 0.2 * np.random.random((sample_len, hid_num)) - 0.1
#     w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1
#     hid_offset = np.zeros(hid_num)
#     out_offset = np.zeros(out_num)
#     input_learnrate = 0.02
#     hid_learnrate = 0.02
#     delt = 1000
#     while delt > 0:
#         for i in range(0, len(sample)):
#             t_label = np.zeros(out_num)
#             t_label[label[i]] = 1
#             # 前向的过程
#             hid_value = np.dot(sample[i], w1)+hid_offset  # 隐层的输入
#             hid_act = get(hid_value)  # 隐层对应的输出
#             out_value = np.dot(hid_act, w2)+out_offset
#             out_act = get(out_value)  # 输出层最后的输出
#
#             # 后向过程
#             err = t_label-out_act
#             out_delta = err*out_act*(1-out_act)  # 输出层的方向梯度方向
#             hid_delta = hid_act*(1 - hid_act) * np.dot(w2, out_delta)
#             for j in range(0, out_num):
#                 w2[:, j] += hid_learnrate*out_delta[j]*hid_act
#             for k in range(0, hid_num):
#                 w1[:, k] += input_learnrate*hid_delta[k]*sample[i]
#
#             out_offset += hid_learnrate * out_delta  # 阈值的更新
#             hid_offset += input_learnrate * hid_delta
#         if delt % 50 == 0:
#             print(sum(err ** 2))
#         delt -= 1
#     return w1, w2, hid_offset, out_offset
#
#
# # 测试过程
# if __name__ == '__main__':
#     train_sample = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     train_label = np.array([0, 1, 1, 1])
#     w1, w2, hid_offset, out_offset = TrainNetwork(train_sample, train_label)
#     right = np.zeros(10)
#     numbers = np.zeros(10)
#     for i in range(len(train_label)):
#         hid_value = np.dot(train_sample[i], w1)+hid_offset
#         hid_act = get(hid_value)
#         out_value = np.dot(hid_act, w2)+out_offset
#         out_act = get(out_value)
#         print(train_sample[i], ":", np.argmax(out_act))
#         if np.argmax(out_act) == train_label[i]:
#             right[train_label[i]] += 1
#     print(right.sum() / len(train_label))
'''
time:20170517
@author: wang xiuxiu
'''
# coding=utf-8
# 反向传播神经网络
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random

random.seed(0)


def rand(a, b):
    """
    创建一个满足 a <= rand < b 的随机数
    :param a:
    :param b:
    :return:
    """
    return (b - a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    """
    创建一个矩阵（可以考虑用NumPy来加速）
    :param I: 行数
    :param J: 列数
    :param fill: 填充元素的值
    :return:
    """
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def randomizeMatrix(matrix, a, b):
    """
    随机初始化矩阵
    :param matrix:
    :param a:
    :param b:
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)


def sigmoid(x):
    """
    sigmoid 函数，1/(1+e^-x)
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    """
    sigmoid 函数的导数
    :param y:
    :return:
    """
    return y * (1 - y)


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        """
        构造神经网络
        :param ni:输入单元数量
        :param nh:隐藏单元数量
        :param no:输出单元数量
        """
        self.ni = ni + 1  # +1 是为了偏置节点
        self.nh = nh
        self.no = no

        # 激活值（输出值）
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 权重矩阵
        self.wi = makeMatrix(self.ni, self.nh)  # 输入层到隐藏层
        self.wo = makeMatrix(self.nh, self.no)  # 隐藏层到输出层
        # 将权重矩阵随机化
        # randomizeMatrix(self.wi, -0.2, 0.2)
        # randomizeMatrix(self.wo, -2.0, 2.0)
        # 权重矩阵的上次梯度
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def runNN(self, inputs):
        """
        前向传播进行分类
        :param inputs:输入
        :return:类别
        """
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]
        # print("ai:", self.ai)
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += ( self.ai[i] * self.wi[i][j] )
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += ( self.ah[j] * self.wo[j][k] )
            self.ao[k] = sigmoid(sum)
        # print(self.ao)
        return self.ao


    def backPropagate(self, targets, N, M):
        """
        后向传播算法
        :param targets: 实例的类别
        :param N: 本次学习率
        :param M: 上次学习率
        :return: 最终的误差平方和的一半
        """
        # http://www.youtube.com/watch?v=aVId8KMsdUU&feature=BFa&list=LLldMCkmXl4j9_v0HeKdNcRA

        # 计算输出层 deltas
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])
        # print(output_deltas)
        # 更新输出层权值
        tmp = []
        for j in range(self.nh):
            for k in range(self.no):
                # output_deltas[k] * self.ah[j] 才是 dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                tmp.append(change)
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change
        # print(tmp)
        # 计算隐藏层 deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])
        # print(hidden_deltas)
        # 更新输入层权值
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change
        # print(self.ci)
        # print(self.wi)
        # 计算误差平方和
        # 1/2 是为了好看，**2 是平方
        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        # print(error)
        return error


    def weights(self):
        """
        打印权值矩阵
        """
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
        print('')

    def test(self, patterns):
        """
        测试
        :param patterns:测试数据
        """
        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1])

    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1):
        """
        训练
        :param patterns:训练集
        :param max_iterations:最大迭代次数
        :param N:本次学习率
        :param M:上次学习率
        """
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                # print("inputs:", inputs)
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            if i % 50 == 0:
                print('Combined error', error)
        self.test(patterns)


def main():
    pat = [
        [[0, 0], [1, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 1], [0, 1]]
    ]
    myNN = NN(2, 3, 2)
    myNN.train(pat)
    myNN.weights()


if __name__ == "__main__":
    main()