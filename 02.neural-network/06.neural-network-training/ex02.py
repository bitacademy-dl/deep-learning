# 신경망학습: 교차 엔트로피 손실함수(Cross Entropy Error, CEE)

import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error_non_batch
except ImportError:
    print('Library Module Can Not Found')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.])
y2 = np.array([0.1, 0.05, 0., 0.4, 0.02, 0.03, 0.1, 0.3, 0., 0.])
y3 = np.array([0., 0.92, 0.02, 0., 0.02, 0.03, 0.01, 0., 0., 0.])


# test
print(cross_entropy_error_non_batch(y1, t))
print(cross_entropy_error_non_batch(y2, t))
print(cross_entropy_error_non_batch(y3, t))
