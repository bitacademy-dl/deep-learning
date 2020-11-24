# 3층 신경망 신호 전달 구현3: 은닉 2층 전달
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(os.getcwd()))
    from ex02 import z1
except ImportError:
    print('Library Module Can Not Found')

print('\n= 신호 전달 구현3: 은닉 2층 전달 ======================')
print(f'z1 dimension: {z1.shape}')  # 3 vector

w2 = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])
print(f'w2 dimension: {w2.shape}')  # 3 x 2 matrix

b2 = np.array([0.1, 0.2])
print(f'b2 dimension: {b2.shape}')  # 2 vector

a2 = np.dot(z1, w2) + b2
print(f'a2 = {a2}')


