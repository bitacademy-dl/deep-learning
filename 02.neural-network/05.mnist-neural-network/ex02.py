# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 신호전달 I
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
except ImportError:
    print('Library Module Can Not Found')

# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()

# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

xlen = len(test_x)
randidx = np.random.randint(0, xlen, 1).reshape(())

# 3. 신호전달
print('\n== 신호전달 구현1: 은닉 1층 전달 ============================')

x = test_x[randidx]
print(f'x dimension: {x.shape}')        # 784 vector

w1 = network['W1']
print(f'w1 dimension: {w1.shape}')      # 784 x 50 matrix
b1 = network['b1']
print(f'b1 dimension: {b1.shape}')      # 50 vector





