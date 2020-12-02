# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet2
# model fitting(training)
import os
import pickle
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

# 1. load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. hyperparameters
batch_size = 100
epochs = 30
learning_rate = 0.1

# 3. model frame config
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1])

# 4. model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

elapsed = 0
epoch_idx = 0
train_losses = []
train_accuracies = []
test_accuracies = []

for idx in range(1, iterations+1):
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]  # 100 x 784
    train_t_batch = train_t[batch_mask]  # 100 x 10

    # 4-2. gradient
    stime = time.time()  # stopwatch: start
    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
    elapsed += (time.time() - stime)  # stopwatch: end

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # 4-5 accuracy per epoch
    if idx % epoch_size == 0:
        epoch_idx += 1
        train_accuracy = network.accuracy(train_x, train_t)
        train_accuracies.append(train_accuracy)

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append(test_accuracy)

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d} ')
        print(f'{int(idx/epoch_idx)}/{epoch_size} - {elapsed*1000:.3f}ms - loss:{loss:.3f} - accuracy:(train, test)=({train_accuracy:.3f}, {test_accuracy:.3f})')

        elapsed = 0

# 5. fitting history
model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

params_file = os.path.join(model_directory, 'twolayer_params.pkl')
trainloss_file = os.path.join(model_directory, 'twolayer_trainloss.pkl')
trainacc_file = os.path.join(model_directory, 'twolayer_trainacc.pkl')
testacc_file = os.path.join(model_directory, 'twolayer_testacc.pkl')

print(f'\nsave model...')

with open(params_file, 'wb') as f_params, \
        open(trainloss_file, 'wb') as f_trainloss, \
        open(trainacc_file, 'wb') as f_trainacc, \
        open(testacc_file, 'wb') as f_testacc:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_trainloss, -1)
    pickle.dump(train_accuracies, f_trainacc, -1)
    pickle.dump(test_accuracies, f_testacc, -1)

print('done')
