# 3층 신경망 신호 전달 구현5: 출력층 전달
import os
import sys
import numpy as np
try:
    sys.path.append(os.path.join(os.getcwd()))
    from ex04 import z2
except ImportError:
    print('Library Module Can Not Found')

print('\n= 신호 전달 구현5: 출력층 전달 ======================')
print(f'z2 dimension: {z2.shape}')  # 2 vector

w3 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])
print(f'w3 dimension: {w3.shape}')  # 2 x 2 matrix

b3 = np.array([0.1, 0.2])
print(f'b3 dimension: {b3.shape}')  # 2 vector

a3 = np.dot(z2, w3) + b3
print(f'a3 = {a3}')

