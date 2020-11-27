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
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1. load train/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load params dataset trained
params_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_params.pkl')
with open(params_file, 'rb') as f:
    network.params = pickle.load(f)

train_accuracy = network.accuracy(train_x, train_t)
test_accuracy = network.accuracy(test_x, test_t)

print(train_accuracy, test_accuracy)


# train_accuracy와 test_accuracy가 일치하는 것은 Overfitting이 발생하지 않았다는 것이다.
# 학습 중에 1epoch 별로 train/test accuracy를 각각 기록하여 추이를 비교하여야 한다.
# 두 accuracy가 마지막까지 차이가 없는 것이 가장 바람직 하며
# 만약 차이가 나면 차이가 나는 그 시점을 찾아서 학습을 중지 하여야 한다 - early stopping



