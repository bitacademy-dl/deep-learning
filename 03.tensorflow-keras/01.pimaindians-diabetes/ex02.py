# pimaindians diabetes model fitting
# tensor-keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1.load training/test data
dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')
x = np.array(dataset[:, 0:8])
t = np.array(dataset[:, 8])
# print(x.shape, t.shape)

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting config
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=200, batch_size=10)

# 5. train loss
loss = history.history['loss']

# 6. result
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy)=({result[0], result[1]})')

# 7. predict
data = np.array([[13, 106, 72, 54, 0, 36.6, 0.178, 45]])
predict = model.predict(data)
percentage = float(predict[0] * 100)
print(f'\n당뇨 발병 확률:{percentage:.2f}%')

# 8. graph
xlen = np.arange(len(loss))
plt.plot(xlen, loss, marker='.', c='blue', label='loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()




