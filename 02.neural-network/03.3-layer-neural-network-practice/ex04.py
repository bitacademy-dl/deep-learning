# 3층 신경망 신호 전달 구현4: 은닉 2층 활성함수 h() 적용
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from ex03 import a2
    from common import sigmoid
except ImportError:
    print('Library Module Can Not Found')

print('\n= 신호 전달 구현4: 은닉 2층 활성함수 h() 적용 ======================')
print(f'a2 dimension: {a2.shape}')  # 2 vector

z2 = sigmoid(a2)
print(f'z2 = {z2}')
