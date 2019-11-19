# coding=utf-8
'''
@Author: TJUZQC
@since: 2019-11-07 16:36:51
@LastAuthor: TJUZQC
@lastTime: 2019-11-19 13:33:54
@Description: None
@FilePath: \ANN\BP_use_keras.py
'''
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(10, input_dim=1))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(lr=0.3)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
model.summary()
x = np.linspace(-1, 1, 300)
# x = np.transpose([x])
noise = np.random.normal(0, 0.05, x.shape)
# noise = np.transpose([noise])
y = np.sin(2 * x) + np.cos(3 * x) + noise
train_x = x[0:200]
train_y = y[0:200]
test_x = x[200:-1]
test_y = y[200:-1]
model.fit(x, y, epochs=200, verbose=1, use_multiprocessing=True)
pred_y = model.predict(x)
# print(pred_y)
plt.scatter(x, y)
plt.plot(x, pred_y, color='red')
plt.show()
