# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet
import os
import pickle
import sys
import time

import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2.hyperparameters
numiters = 1  # ?
szbatch = 100
sztrain = train_x.shape[0]  # 60,000
ratelearning = 0.1

# 3.initialize network
network.initialize(sz_input=train_x.shape[1], sz_hidden=50, sz_output=train_t.shape[1])

# 4.training
train_losses = []

for idx in range(numiters):

    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    train_x_batch = train_x[batch_mask]                 # 100 x 784
    train_t_batch = train_t[batch_mask]                 # 100 x 10

    # 4-2. gradient
    stime = time.time()             # stopwatch: start
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
    elapsed = time.time() - stime   # stopwatch: end

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    print(f'#{idx+1}: loss:{loss}, elapsed time: {elapsed}s')

# serialize train loss
train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer-train-loss.pkl')
print(f'Save Pickle({train_loss_file}) file....')
with open(train_loss_file, 'wb') as f:
    pickle.dump(train_losses, f, -1)
print('Done!')
