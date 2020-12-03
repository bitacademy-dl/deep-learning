# Iris Species Multi-Class Classification Model(Iris 품종 예측 모델)
# Model Fitting(학습)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

# 1.load training/test data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

dateset_file = './dataset/iris.csv'
df = pd.read_csv(dateset_file, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])

dataset = df.values
x, t = dataset[:, 0:4].astype(float), dataset[:, 4]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=50, batch_size=1, verbose=1)

# 5. result
loss = history.history['loss']
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy)=({result[0]}, {result[1]})')

# 6. predict
data = np.array([[6.4, 2.8, 5.6, 2.2]])
predict = model.predict(data)
index = np.argmax(predict)
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(f'예측 되는 품종은 {species[index]} 입니다.')

# 7. graph
xlen = np.arange(len(loss))
plt.plot(xlen, loss, marker='.', c='blue', label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
