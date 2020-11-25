# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 배치처리
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
    from common import sigmoid, softmax
except ImportError:
    print('Library Module Can Not Found')

# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 3. 정확도 산출
hit = 0
xlen = len(test_x)
batch_size = 100

for idx, batch_sidx in enumerate(range(0, xlen, batch_size)):
    batch_x = test_x[batch_sidx:batch_sidx+batch_size]

    a1 = np.dot(batch_x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    batch_y = softmax(a3)

    batch_predict = np.argmax(batch_y, axis=1)
    # print(batch_predict)

    batch_t = test_t[batch_sidx:batch_sidx+batch_size]
    # print(batch_t)

    batch_hit = np.sum(batch_predict == batch_t)
    hit += batch_hit

    print(f'batch #{idx+1}, batch hit: {batch_hit}, total hit:{hit}')


# 정확도(Accuracy)
print(f'Accuracy: {hit/xlen}')





