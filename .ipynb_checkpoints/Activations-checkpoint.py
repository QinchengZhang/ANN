# coding=utf-8
'''
@Author: TJUZQC
@since: 2019-11-07 16:47:24
@LastAuthor: TJUZQC
@lastTime: 2019-11-19 13:31:46
@Description: None
@FilePath: \ANN\Activations.py
'''
import numpy as np


def sigmoid(x):
    '''
    @description: sigmoid激活函数 
    @param x: 输入x
    @return: x激活后的值
    '''
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_deriv(x):
    '''
    @description: sigmoid激活函数的微分
    @param x: 输入X
    @return: x的sigmoid微分
    '''
    return x * (1-x)


def ReLU(x):
    return (np.abs(x) + x) / 2.0


def ReLU_deriv(x):
    relu = ReLU(x)
    relu[relu <= 0] = 0
    relu[relu > 0] = 1
    return relu

# 双曲函数


def tanh(x):
    return np.tanh(x)

# 双曲函数的微分


def tanh_deriv(x):
    return 1.0 - x ** 2


def linear(x):
    return x


def linear_deriv(x):
    return np.ones_like(x)


def SeLU(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * x if x > 0 else alpha * (np.exp(x) - 1.0) * scale


def SeLU_deriv(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return alpha * scale * np.exp(x) if x < 0 else scale
