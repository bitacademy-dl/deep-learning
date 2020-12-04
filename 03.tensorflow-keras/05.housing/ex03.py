# Housing Price Prediction(Linear Regression) Model(보스톤 집값 예측 모델)
# Model fitting - dl class version
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    import multilayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
dateset_file = './dataset/housing.csv'
df = pd.read_csv(dateset_file, delim_whitespace=True, header=None)
dataset = df.values
x = dataset[:, 0:13]
t = dataset[:, 13]

train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)

train_t = train_t[:, np.newaxis]
train_t = train_t[:, np.newaxis]


# 2. hyperparameters
batch_size = 10
epochs = 200
learning_rate = 0.01

# 3. model frame config
network.initialize(input_size=train_x.shape[1], hidden_size=[30, 10], output_size=train_t.shape[1])

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

elapsed = 0
epoch_idx = 0
for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]  # 100 x 784
    train_t_batch = train_t[batch_mask]  # 100 x 10

    # 4-2. gradient
    stime = time.time()  # stopwatch: start
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
    elapsed += (time.time() - stime)  # stopwatch: end

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)

    # 4-5 accuracy per epoch
    if idx % epoch_size == 0:
        epoch_idx += 1

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d} ')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000:.3f}ms - loss:{loss:.3f}')

        elapsed = 0





