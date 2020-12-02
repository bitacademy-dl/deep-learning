# pimaindians diabetes model fitting
# TwoLayerNet2
import numpy as np

# 1.load training/test data
dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')
train_x = np.array(dataset[:, 0:8])

train_t = np.array(dataset[:, 8])
# print(train_t)
train_t = train_t[:, np.newaxis]
# print(train_t)

train_t = np.c_[train_t, train_t == 0]
print(train_t)
