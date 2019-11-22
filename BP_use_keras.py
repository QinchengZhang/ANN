# coding=utf-8
'''
@Author: TJUZQC
@since: 2019-11-07 16:36:51
@LastAuthor: TJUZQC
@lastTime: 2019-11-22 15:44:52
@Description: None
@FilePath: \ANN\BP_use_keras.py
'''
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

def genrate_actual_label(x):
    noise = np.random.normal(0, 0.05, x.shape)
    return np.sin(2 * x) + np.cos(x) + noise

model = keras.Sequential()
model.add(keras.layers.Dense(20, input_dim=1))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(lr=0.025)
model.compile(optimizer='Nadam', loss='mse', metrics=['accuracy'])
model.summary()
x = np.linspace(-np.pi, 2 * np.pi, 600)
y = genrate_actual_label(x)
train_x = np.array(x[0:-1])
train_y = np.array(y[0:-1])
x = np.linspace(-np.pi, 2 * np.pi, 600)
y = genrate_actual_label(x)
test_x = np.array(x[0:-1])
test_y = np.array(y[0:-1])
model.fit(train_x, train_y, epochs=200, verbose=1, use_multiprocessing=True)
pred_y = model.predict(test_x)
# print(pred_y)
plt.scatter(test_x, test_y)
plt.plot(test_x, pred_y, color='red')
plt.show()
