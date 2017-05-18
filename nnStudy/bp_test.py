'''
time:20170516
@author: wang xiuxiu
'''
import numpy as np
import scipy.io as sio  # 为了导入mat文件
from sklearn import datasets
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
np.random.seed(1)


# 导入moons数据
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def tanh(x):
    return np.tanh(x)


# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):
    return 1.0 - tanh(x) ** 2


# sigmoid函数
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# sigmoid求导结果
def de_sigmoid(y):
    return y*(1 - y)


st = time.clock()
class NN_BP:
    # 三层反向传播神经网络
    def __init__(self, ni, no, nh):
        # 节点的数目
        self.ni = ni  # 偏置
        self.no = no
        self.nh = nh
        #
        # # 激活节点
        # self.ai = np.ones((1, self.ni))
        # self.ao = np.ones((1, self.no))
        # # print(self.ao.shape)
        # self.ah = np.ones((1, self.nh))

        # 权重矩阵
        self.w_i2h = 4*np.random.random((self.ni, self.nh))-2
        self.w_h2o = 4*np.random.random((self.nh, self.no))-2
        # self.w_i2h = np.ones((self.ni, self.nh))
        self.b1 = np.zeros((1, self.nh))
        # self.w_h2o = np.ones((self.nh, self.no))
        self.b2 = np.zeros((1, self.no))
        # 动量因子
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))
    # 初始化时间
    print("init of time:", st-time.clock())
    # 计算，并获得隐藏层和输出节点的值
    def update(self, inputs, targets, l1, l2):
        # 正向传播
        ai = inputs  # 输入节点
        ah = sigmoid(np.dot(ai, self.w_i2h) + self.b1)
        ao = sigmoid(np.dot(ah, self.w_h2o) + self.b2)
        # print("ao:", ao)
        # 反向传播
        # 计算误差
        # 按照输出节点补齐targets
        if targets.size > 1:
            t_label = np.zeros((1, self.no))
            t_label[0, targets] = 1
            error = t_label - ao
        else:
            error = targets - ao

        output_deltas = de_sigmoid(ao) * error  # 1 * no
        change_out = np.dot(ah.T, output_deltas)  # 更改值 nh*no，ah为1*nh
        # print(change_out)
        self.w_h2o = self.w_h2o + l1 * change_out + l2 * self.co
        self.co = change_out
        self.b2 += l1 * output_deltas

        er_hid = np.dot(self.w_h2o, output_deltas.T)  # 1 * nh
        hidden_delta = er_hid * de_sigmoid(ah).T  # 1 * nh
        change = np.dot(ai.T, hidden_delta.T)  # ni * nh

        self.w_i2h = self.w_i2h + l1 * change + l2 * self.ci
        self.b1 += l1 * hidden_delta.T
        self.ci = change

        # 计算误差平方和
        # 1/2 是为了好看，**2 是平方
        error = 0.5 * sum(error ** 2)
        return error

    def weights(self):
        """
        打印权值矩阵
        """
        print('Input weights:')
        for i in range(self.ni):
            print(self.w_i2h[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.w_h2o[j])
        print('')

    def test(self, inputs, targets):
        """
        测试
        :param patterns:测试数据
        """
        for j in range(inputs.shape[0]):
            # 正向传播
            ai = np.array(inputs[j])  # 输入节点
            ah = sigmoid(np.dot(ai, self.w_i2h) + self.b1)
            ao = sigmoid(np.dot(ah, self.w_h2o) + self.b2)
            print('Inputs:', inputs[j], '-->', ao, '\tTarget', targets[j])

    def train(self, inputs, targets, max_iterations=1000, l1=0.5, l2=0.01):
        """
        训练
        :param patterns:训练集
        :param max_iterations:最大迭代次数
        :param N:本次学习率
        :param M:上次学习率
        """
        for i in range(max_iterations):
            for j in range(inputs.shape[0]):
                # print(inputs.shape)
                # print(targets.size)
                error = self.update(np.array([inputs[j]]), targets[j], l1, l2)

            if i % 50 == 0:
                print('Combined error', error)
        self.test(inputs, targets)

    def predict(self, x):
        res = []
        for j in range(x.shape[0]):
            ai = np.array(x[j])  # 输入节点
            ah = sigmoid(np.dot(ai, self.w_i2h) + self.b1)
            ao = sigmoid(np.dot(ah, self.w_h2o) + self.b2)
            res.append(np.argmax(ao))
        return res


def main(train_data, train_label, test_data, test_label):
    # data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # label = np.array([1, 0, 0, 1])
    # label = np.array([[0, 1], [1, 0], [1, 0], [1,0]])
    # data, label = generate_data()
    # train_data = np.array(data)
    # train_label = np.array(label)

    # 手写体
    myNN = NN_BP(784, 10, 15)
    myNN.train(train_data, train_label)

    # predict
    res = myNN.predict(test_data)
    right = 0
    if res == test_label:
        right += 1
    print(right)
    print(classification_report(y_pred=res, y_true=res))
    # plot_decision_boundary(lambda x: myNN.predict(x), train_data, label)
    # plt.title("Logistic Regression")


if __name__ == "__main__":
    # main()
    train_ = sio.loadmat("./BPNetwork-master/mnist_train.mat")
    train_ = train_["mnist_train"]
    train_label = sio.loadmat("./BPNetwork-master/mnist_train_labels.mat")
    train_label = np.array(train_label["mnist_train_labels"])
    # print(train_label.shape)
    main(train_[0:100, :], train_label[0:100, 0], train_[101:200, :], train_label[101, 0])

    print("count of time:", st-time.clock())