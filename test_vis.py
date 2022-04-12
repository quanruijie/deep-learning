# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:49:19 2022
@author: ruijie
"""
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

with open('model.pickle','rb') as file:
    model = pickle.load(file)

    trainer = Trainer(model, x_train, t_train, x_test, t_test,
                      epochs=50, mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr': 0.0245})
    trainer.train()
    
    # return trainer.test_acc_list, trainer.train_acc_list, \
    #         trainer.test_loss_list, trainer.train_loss_list, network
    
    print(trainer.test_acc_list[-1])
    
    plt.subplot(121)
    plt.plot(trainer.test_acc_list, 'c')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.subplot(122)
    plt.plot(trainer.test_loss_list, 'm')
    plt.ylabel('loss')
    plt.title('test loss')
    
    plt.savefig('test_accuracy_VS_loss.png')
    plt.show()
    
    # 可视化参数
    plt.subplot(121)
    plt.imshow(model.params['W1'])
    plt.subplot(122)
    plt.imshow(model.params['W2'])
    plt.savefig('weight.png')
    plt.show()
    
    file.close()
