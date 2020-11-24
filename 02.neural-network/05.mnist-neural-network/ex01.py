# MNIST(Modified National Institude of Standard and Technology)
# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 데이터 살펴보기

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
except ImportError:
    print('Library Module Can Not Found')


(train_x, train_t), (test_x, test_t) = load_mnist(normalize=False, flatten=True, one_hot_label=False)