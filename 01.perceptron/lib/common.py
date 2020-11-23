import numpy as np


# step 활성함수
def step(x):
    return np.array(x > 0, dtype=np.int)