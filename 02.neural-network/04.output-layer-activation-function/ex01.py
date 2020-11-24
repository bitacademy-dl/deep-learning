# 출력함수(출력층 활성함수) σ() - 항등함수(Identity Function)
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity
except ImportError:
    print('Library Module Can Not Found')

x = np.arange(-10, 10, 0.1)
y = identity(x)

plt.plot(x, y)
plt.show()
