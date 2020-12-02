# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet
# Estimation: Training Accuracy
import os
import pickle
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet2 as network
except ImportError:
    print('Library Module Can Not Found')

# 1. load train/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load params dataset trained
params_file = os.path.join(os.getcwd(), 'model', 'twolayer_params.pkl')
params = None
with open(params_file, 'rb') as f:
    params = pickle.load(f)

# 3. model frame
network.initialize(input_size=train_x.shape[1], hidden_size=50, output_size=train_t.shape[1], init_params=params)

train_accuracy = network.accuracy(train_x, train_t)
test_accuracy = network.accuracy(test_x, test_t)

print(train_accuracy, test_accuracy)


