# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-11-07 16:47:24
@LastEditors: TJUZQC
@LastEditTime: 2019-12-10 16:53:27
@Description: None
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
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

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


