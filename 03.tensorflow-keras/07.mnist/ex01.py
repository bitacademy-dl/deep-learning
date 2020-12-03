# MNIST handwritten digit classification model
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
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
history = model.fit(train_x, train_t, validation_data=(test_x, test_t), epochs=30, batch_size=100, verbose=1)

# 5. train loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']     # validation loss(test loss)

# 6. graph
xlen = np.arange(len(train_loss))

plt.plot(xlen, train_loss, marker='.', c='blue', label='Train Loss')
plt.plot(xlen, test_loss, marker='.', c='red', label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.show()