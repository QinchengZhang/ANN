# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-11-22 18:18:53
@LastAuthor: TJUZQC
@lastTime: 2019-11-22 19:07:05
@Description: None
@FilePath: \ANN\hdf5_test.py
'''
import h5py
import numpy as np
import Model

# f = h5py.File('BPNN.h5','r')
# print(f['activation_hidden'][:])
model = Model.ANN.load('BPNN.h5')
# model = Model.BPNN([1,4,1])
# model.save('BPNN.h5')