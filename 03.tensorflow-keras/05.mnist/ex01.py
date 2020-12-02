# MNIST handwritten digit classification model
import os
import sys

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

try:
    sys.path.append(os.path.join(os.getcwd(), 'lib'))
    from mnist import load_mnist
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(train_t.shape[1], activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. model fitting
model.fit(train_x, train_t, epochs=30, batch_size=100, verbose=1)
