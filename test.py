# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-11-19 10:51:38
@LastAuthor: TJUZQC
@lastTime: 2019-11-22 18:06:00
@Description: None
@FilePath: \ANN\train.py
'''
import numpy as np
from Model import BPNN
from matplotlib import pyplot as plt

def genrate_actual_label(x):
    noise = np.random.normal(0, 0.05, x.shape)
    return np.sin(2 * x) + np.cos(x) + noise


x = np.linspace(-np.pi, 2 * np.pi, 600)
x = np.transpose([x])
y = genrate_actual_label(x)
train_x = np.array(x[0:-1])
train_y = np.array(y[0:-1])
x = np.linspace(-np.pi, 2 * np.pi, 600)
x = np.transpose([x])
y = genrate_actual_label(x)
test_x = np.array(x[0:-1])
test_y = np.array(y[0:-1])
model = BPNN([1, 2, 1], activation_hidden='tanh', activation_out='linear')
# model.load_weights('BPNN.npy')
model.forward_propagation(np.array([[1],[2]]))
model.back_propagation(np.array([[1],[2]]))
