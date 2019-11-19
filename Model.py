# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-11-12 15:40:52
@LastAuthor: TJUZQC
@lastTime: 2019-11-19 13:37:45
@Description: None
@FilePath: \ANN\Model.py
'''
import Activations
import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import random

np.set_printoptions(precision=6)


class BPNN():
    def __init__(self, layers, activation_hidden='sigmoid', activation_out='linear'):
        '''
        @description: 构造函数
        @param layers: 一个list实例,eg:[1,4,1]为三层,输入层1个节点,隐藏层4节点,输出层1节点
        @param activation_hidden: 输出层之前的激活函数,不区分大小写,默认为sigmoid
        @param activation_out: 输出层的激活函数,不区分大小写,默认为linear
        @return: 无
        '''
        self.activation_hidden, self.activation_hidden_deriv = self.init_activation(
            activation_hidden)
        self.activation_out, self.activation_out_deriv = self.init_activation(
            activation_out)
        self.weights = self.init_weights(layers)
        self.bias = self.init_bias(layers)
        self.deltas = []
        self.layers_in = []
        self.layers_out = []
        self.learning_rate = 0.1

    def init_activation(self, activation):
        '''
        @description: 激活函数初始化
        @param activation: 激活函数名,不区分大小写 
        @return: 激活函数, 激活函数的导函数
        '''
        activations = ['sigmoid', 'relu', 'tanh', 'linear', 'selu']
        if activation.lower() == 'sigmoid':
            return Activations.sigmoid, Activations.sigmoid_deriv
        elif activation.lower() == 'relu':
            return Activations.ReLU, Activations.ReLU_deriv
        elif activation.lower() == 'tanh':
            return Activations.tanh, Activations.tanh_deriv
        elif activation.lower() == 'linear':
            return Activations.linear, Activations.linear_deriv
        elif activation.lower() == 'selu':
            return Activations.SeLU, Activations.SeLU_deriv
        else:
            raise ValueError(
                'this activate function does not supoorted, now supported activations are {}'.format(activations))

    def init_weights(self, layers):
        '''
        @description: 初始化权重
        @param layers: 一个list实例,eg:[1,4,1]为三层,输入层1个节点,隐藏层4节点,输出层1节点
        @return: 权重数组
        '''
        weights = []
        for i in range(1, len(layers)):
            weights.append(np.random.normal(0, 0.25, (layers[i-1], layers[i])))
        return weights

    def init_bias(self, layers):
        '''
        @description: 初始化偏置项
        @param layers: 一个list实例,eg:[1,4,1]为三层,输入层1个节点,隐藏层4节点,输出层1节点
        @return: 偏置项数组
        '''
        bias = []
        for i in range(1, len(layers)):
            bias.append(np.random.normal(0, 0.25, (1, layers[i])))
        return bias

    def forward_propagation(self, x):
        '''
        @description: 前向传播
        @param x: 输入x
        @return: 经过神经网络传播后的值
        '''
        self.layers_in = [np.array([x])]
        self.layers_out = [self.layers_in[-1]]
        for i in range(len(self.weights)-1):
            self.layers_in.append(
                np.array(np.dot(self.layers_out[-1], self.weights[i]) + self.bias[i]))
            self.layers_out.append(self.activation_hidden(self.layers_in[-1]))
        self.layers_in.append(
            np.array(np.dot(self.layers_out[-1], self.weights[-1]) + self.bias[-1]))
        self.layers_out.append(self.activation_out(self.layers_in[-1]))
        return self.layers_out[-1][0][0]

    def back_propagation(self, actual_label):
        '''
        @description: 反向传播
        @param actual_label: 真实值标签
        @return: 精确度, 损失
        '''
        self.deltas = [np.dot(self.activation_out_deriv(self.layers_out[-1]),
                              -(actual_label - self.layers_out[-1]))]
        for i in range(len(self.weights) - 1, 0, -1):
            self.deltas.append(np.dot(np.diag(self.activation_hidden_deriv(self.layers_out[i])[
                               0]), np.dot(self.weights[i], self.deltas[-1])))
        self.deltas.reverse()
        loss = self.cal_loss(actual_label[0], self.layers_out[-1][0][0])
        accuracy = self.cal_accuracy(
            actual_label[0], self.layers_out[-1][0][0])
        return accuracy, loss

    def update_parameters(self):
        '''
        @description: 更新参数
        @return: 无
        '''
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * \
                (np.dot(self.deltas[i], self.layers_out[i]).T)

            self.bias[i] -= self.learning_rate * self.deltas[i].T

    def fit(self, train_data, train_label, learning_rate=0.1, epochs=10):
        '''
        @description: 开始训练
        @param train_data: 训练集的输入x
        @param train_label: 训练集的真实值y
        @param learning_rate: 学习率
        @param epochs: 训练的次数
        @return: 无
        '''
        self.learning_rate = learning_rate
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            outputs = []
            i = np.random.randint(train_data.shape[0])
            outputs.append(self.forward_propagation(train_data[i]))
            accuracy, loss = self.back_propagation(train_label[i])
            pbar.set_description(
                'epoch {}: acc:{:.3f}, loss:{:.3f}'.format(epoch+1, accuracy, loss))
            self.update_parameters()
            time.sleep(0.001)

    def predict(self, test_data):
        '''
        @description: 开始预测
        @param test_data: 测试集的输入x 
        @return: 预测的输出y`
        '''
        pred_y = []
        for i in range(len(test_data)):
            # print('test_data[{}] = {}'.format(i, test_data[i]))
            pred_y.append(self.forward_propagation(test_data[i]))
        return pred_y

    def save_weights(self, filename: str):
        '''
        @description: 保存权重
        @param filename: npy文件名
        @return: 无
        '''
        parameters = np.array([self.weights, self.bias])
        if filename.split('.')[-1] == 'npy':
            np.save(filename, parameters)
        else:
            np.save('{}.npy'.format(filename), parameters)

    def load_weights(self, filename: str):
        '''
        @description: 加载权重
        @param filename: npy文件名
        @return: 无
        '''
        parameters = np.load(filename, allow_pickle=True)
        weights = parameters[0]
        bias = parameters[1]
        if weights.shape[0] != len(self.weights) or bias.shape[0] != len(self.bias):
            raise ValueError('layers\' shape must be same!')
        else:
            self.weights = weights
            self.bias = bias

    def cal_accuracy(self, actual_label, output):
        '''
        @description: 计算精确度
        @param actual_label: 真实标签y
        @param output: 前向传播得到的输出y`
        @return: 精确度
        '''
        return 1.0 - abs(abs(output - actual_label)/actual_label)

    def cal_loss(self, actual_label, output):
        '''
        @description: 计算损失
        @param actual_label: 真实标签y
        @param output: 前向传播得到的输出y`
        @return: 损失
        '''
        return 0.5 * pow(actual_label - output, 2)

    def accuracy_calculation(self, actual_label, output):
        '''
        @description: 计算准确率
        @param actual_label: 真实标签y
        @param output: 前向传播得到的输出y`
        @return: 准确率（百分制）
        '''
        correct_count = 0
        for i in range(len(actual_label)):
            if actual_label[i] == output[i]:
                correct_count += 1
        return correct_count / float(len(actual_label)) * 100.0
