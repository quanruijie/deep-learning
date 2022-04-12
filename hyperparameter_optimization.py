# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from util import shuffle_dataset
from trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了实现高速化，减少训练数据
x_train = x_train[:1000]
t_train = t_train[:1000]

# 分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(hidden, lr, weight_decay, epocs=50):
    network = TwoLayerNet(input_size = 784, hidden_size = hidden, output_size = 10,
                          lmd = weight_decay, use_dropout = False, dropout_ration = 0.1)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr': lr})
    trainer.train()
    
    return trainer.test_acc_list, trainer.train_acc_list, \
            trainer.test_loss_list, trainer.train_loss_list, network


##搜索超参数======================================
results_val_acc = {}
results_train_acc = {}
results_val_loss = {}
results_train_loss = {}

# for hidden in {50}: # range(50, 70, 10):
#     for lr in np.linspace(0.02, 0.03, 30):      # 50 0.0224 0
#         for lmd in 10 ** np.linspace(-6, -5, 2):
#             val_acc, train_acc, val_loss, train_loss, network = __train(hidden, lr, lmd)
#             # print("val acc:" + str(round(val_acc[-1], 4)) + 
#             #       "  \t| hidden:" + str(hidden) + 
#             #       ", lr:" + str(round(lr, 4)) + 
#             #       ", weight decay:" + str(round(lmd, 4)))
#             key = "hidden:" + str(hidden) + ", lr:" + str(round(lr, 4)) + \
#                     ", weight decay:" + str(round(lmd, 6))
#             results_val_acc[key] = val_acc
#             results_train_acc[key] = train_acc
            
#             results_val_loss[key] = val_loss
#             results_train_loss[key] = train_loss


# val_acc, train_acc, val_loss, train_loss, _ = __train(50, 0.01, 1e-6)
# print(val_acc[-1], val_loss[-1])



# 绘制图形========================================================
# print("=========== Hyper-Parameter Optimization Result ===========")
# graph_draw_num = 12
# col_num = 4
# row_num = int(np.ceil(graph_draw_num / col_num))
# i = 0

# for key, val_acc_list in sorted(results_val_acc.items(), key=lambda x:x[1][-1], reverse=True):
#     print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") |\t" + key)

#     plt.subplot(row_num, col_num, i+1)
#     plt.title("Best-acc" + str(i+1))
#     plt.ylim(0.0, 1.0)
#     if i % 4: 
#         plt.yticks([])
#         plt.xticks([])
#     x = np.arange(len(val_acc_list))
#     plt.plot(x, val_acc_list, label = "val_acc")
#     plt.plot(x, results_train_acc[key], "--", label = "train_acc")
#     i += 1

#     if i >= graph_draw_num:
#         break
# plt.legend()
# plt.savefig("Best_12_accuracy.png")
# plt.show()

############### print(loss)######

# i = 0
# for key, val_loss_list in sorted(results_val_loss.items(), key=lambda x:x[1][-1]):
#     print("Best-" + str(i+1) + "(val loss:" + str(round(val_loss_list[-1], 4)) + ") |\t" + key)

#     plt.subplot(row_num, col_num, i+1)
#     plt.title("Best-loss" + str(i+1))
#     plt.ylim(0.0, 1.0)
#     if i % 4: 
#         plt.yticks([])
#         plt.xticks([])
#     x = np.arange(len(val_loss_list))
#     plt.plot(x, val_loss_list, label = "val_loss")
#     plt.plot(x, results_train_loss[key], "--", label = "train_loss")
#     i += 1

#     if i >= graph_draw_num:
#         break
# plt.legend()
# plt.savefig("Best_12_loss.png")
# plt.show()


bestModel = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10,
                          lmd = 1e-6, use_dropout = False, dropout_ration = 0.1)
bestModel.save("model.pickle")
