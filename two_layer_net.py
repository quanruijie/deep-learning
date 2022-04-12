# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from layers import *
from collections import OrderedDict
import pickle


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, 
                 lmd = 0, weight_init_std = 0.01, use_dropout = False, dropout_ration = 0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lmd = lmd
        self.use_dropout = use_dropout
        
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        if self.use_dropout:
            self.layers['dropout'] = Dropout(dropout_ration)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        reg = 0
        reg += 0.5 * np.sum(self.params['W1'] ** 2) * self.lmd
        reg += 0.5 * np.sum(self.params['W2'] ** 2) * self.lmd
        
        return self.lastLayer.forward(y, t) + reg
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : 
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW + self.lmd * self.layers['Affine1'].W
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW + self.lmd * self.layers['Affine2'].W
        grads['b2'] = self.layers['Affine2'].db

        return grads
    
    def save(self,fname):
        f = open(fname,'wb')
        pickle.dump(self,f)
        f.close()
