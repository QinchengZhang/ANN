# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-11-19 10:51:38
@LastAuthor: TJUZQC
@lastTime: 2019-11-19 13:38:02
@Description: None
@FilePath: \ANN\train.py
'''
import numpy as np
from Model import BPNN
from matplotlib import pyplot as plt


def genrate_actual_label(x):
    noise = np.random.normal(0, 0.05, x.shape)
    return 0.5 * np.sin(x) + 0.5 * np.cos(x) + noise


x = np.linspace(-np.pi, np.pi, 300)
x = np.transpose([x])
y = genrate_actual_label(x)
train_x = np.array(x[0:-1])
train_y = np.array(y[0:-1])
x = np.linspace(-np.pi, np.pi, 300)
x = np.transpose([x])
y = genrate_actual_label(x)
test_x = np.array(x[0:-1])
test_y = np.array(y[0:-1])
model = BPNN([1, 10, 10, 1], activation_hidden='tanh', activation_out='linear')
# model.load_weights('BPNN.npy')
model.fit(train_x, train_y, epochs=4000, learning_rate=0.3)
model.save_weights('BPNN.npy')
y_pred = model.predict(test_x)
plt.scatter(test_x, test_y, label='GT')
plt.plot(test_x, y_pred, color='red', label='predict')
plt.legend()
plt.show()
