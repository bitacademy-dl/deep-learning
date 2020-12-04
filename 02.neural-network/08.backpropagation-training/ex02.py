# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: MultiLayerNet
# Test: SGD based on Backpropagation Gradient
import os
import sys
import time
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet2 as network
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2.hyperparameters
iterations = 1  # 12000
batch_size = 100
train_size = train_x.shape[0]
learning_rate = 0.1

# 3.initialize network
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1])

# 4.training
train_losses = []

for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]                 # 100 x 784
    train_t_batch = train_t[batch_mask]                 # 100 x 10

    # 4-2. gradient
    stime = time.time()             # stopwatch: start
    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
    elapsed = time.time() - stime   # stopwatch: end

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    print(f'#{idx}: loss:{loss:.3f}, elapsed time: {elapsed*1000:.3f}ms')


