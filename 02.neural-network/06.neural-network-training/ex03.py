# 신경망학습: 미니 배치(mini-batch)
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import cross_entropy_error
except ImportError:
    print('Library Module Can Not Found')

# test1
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
print(train_x.shape)    # 60000 x 784
print(train_t.shape)    # 60000 x 10

train_size = len(train_x)
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)

train_x_batch = train_x[batch_mask]
train_t_batch = train_t[batch_mask]
print(train_x_batch.shape)  # 10 X 784
print(train_t_batch.shape)  # 10 x 10

# test3
# 만약에 batch_size가 3인 경우
t = np.array([
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
])

y = np.array([
    [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.],
    [0.1, 0.05, 0., 0.4, 0.02, 0.03, 0.1, 0.3, 0., 0.],
    [0., 0.92, 0.02, 0., 0.02, 0.03, 0.01, 0., 0., 0.]
])


print(cross_entropy_error(y, t))



