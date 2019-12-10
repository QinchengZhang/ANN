# coding=utf-8
'''
@Author: TJUZQC
@Date: 2019-12-10 17:03:14
@LastEditors: TJUZQC
@LastEditTime: 2019-12-10 17:06:35
@Description: None
'''
def MSE(y_true, y):
    return 0.5 * pow(y_true - y, 2)

def MSE_deriv(y_true, y):
    return -(y_true - y)