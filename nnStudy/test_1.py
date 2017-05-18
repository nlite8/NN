'''
time:20170511
@author: wang xiuxiu
'''

import numpy as np
import math


# 定义Sigmoid函数
def get(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec


# 训练BP神经网络
def TrainNetwork(sample, label):
    sample_num = len(sample)
    sample_len = len(sample[0])
    out_num = 10
    hid_num = 8
    w1 = 0.2 * np.random.random((sample_len, hid_num)) - 0.1
    w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1
    hid_offset = np.zeros(hid_num)
    out_offset = np.zeros(out_num)
    input_learnrate = 0.2
    hid_learnrate = 0.2
    for i in range(0, len(sample)):
        t_label = np.zeros(out_num)
        t_label[label[i]] = 1
        # 前向的过程
        hid_value = np.dot(sample[i], w1)+hid_offset  # 隐层的输入
        hid_act = get(hid_value)  # 隐层对应的输出
        out_value = np.dot(hid_act, w2)+out_offset
        out_act = get(out_value)  # 输出层最后的输出

        # 后向过程
        err = t_label-out_act
        out_delta = err*out_act*(1-out_act)  # 输出层的方向梯度方向
        hid_delta = hid_act*(1 - hid_act) * np.dot(w2, out_delta)
        for j in range(0, out_num):
            w2[:, j] += hid_learnrate*out_delta[j]*hid_act
        for k in range(0, hid_num):
            w1[:, k] += input_learnrate*hid_delta[k]*sample[i]

        out_offset += hid_learnrate * out_delta  # 阈值的更新
        hid_offset += input_learnrate * hid_delta

    return w1, w2, hid_offset, out_offset


# 测试过程
if __name__ == '__main__':
    train_sample = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_label = np.array([0, 1, 1, 1])
    w1, w2, hid_offset, out_offset = TrainNetwork(train_sample, train_label)
    right = np.zeros(10)
    numbers = np.zeros(10)

# import numpy as np
#
# X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# Y = np.array([[0, 0, 1, 1]]).T
#
# syn0 = 2*np.random.random((3, 4)) - 1
# syn1 = 2*np.random.random((4, 1)) - 1
#
# # obj: y = syn0*X + syn1
# # .T是转置的意思
# for j in range(200):
#     l1 = 1/(1+np.exp(-(np.dot(X, syn0))))  # 4*4
#     # print(l1)
#     l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))  # 4*1
#     # print(l2)
#     l2_delta = -(l2-Y)*(l2*(1-l2))  # 4*1
#     # print(l2_delta)

    # l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    # syn1 += l1.T.dot(l2_delta)
    # syn0 += X.T.dot(l1_delta)
    # if j % 20 == 0:
    #     print(l2_delta)
    #     print(l1_delta)
