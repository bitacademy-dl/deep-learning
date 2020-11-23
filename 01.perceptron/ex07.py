# 계단함수

import os
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Found')


x = np.arange(-5.0, 5.0, 0.1)
y = step(x)

plt.plot(x, y)
plt.show()

