# 3층 신경망 신호 전달 구현6: 출력층 출력함수 σ() 적용
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from ex05 import a3
    from common import identity
except ImportError:
    print('Library Module Can Not Found')

print('\n= 신호 전달 구현6: 출력층 출력함수 σ() 적용 ======================')
print(f'a3 dimension: {a3.shape}')  # 2 vector

y = identity(a3)
print(f'y = {y}')


