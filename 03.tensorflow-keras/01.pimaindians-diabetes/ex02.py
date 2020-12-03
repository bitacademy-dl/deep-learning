# pimaindians diabetes model fitting
# tensor-keras
import numpy as np

# 1.load training/test data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')
x = np.array(dataset[:, 0:8])
t = np.array(dataset[:, 8])
# print(x.shape, t.shape)

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_size=x.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting config


# 4. model fitting


# 5. train loss


# 6. graph


