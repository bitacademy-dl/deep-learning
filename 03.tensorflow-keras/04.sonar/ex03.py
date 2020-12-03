# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Model Fitting #2 - Overfitting
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 1-1. load training/test data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model

dateset_file = './dataset/sonar.csv'
df = pd.read_csv(dateset_file, header=None)

dataset = df.values
x, t = dataset[:, 0:60].astype(float), dataset[:, 60]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
# t = tf.keras.utils.to_categorical(t)

# 1-2. Split (Train & Test) Dataset
train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)
# print(train_x.shape, train_t.shape)
# print(test_x.shape, test_t.shape)

# 2. model frame config
model = Sequential()
model.add(Dense(20, input_dim=60, activation='relu'))
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(1, input_dim=10, activation='sigmoid'))

# 3. model fitting config
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(train_x, train_t, epochs=200, batch_size=5, verbose=1)

# 5. result
loss = history.history['loss']
result = model.evaluate(train_x, train_t, verbose=0)
print(f'\n(Train Loss, Train Accuracy)=({result[0]}, {result[1]})')

# 6. save model
model_directory = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_directory):
    os.mkdir(model_directory)

model.save(os.path.join(model_directory, 'model.h5'))

# 7. test - overfitting
del model
model = load_model(os.path.join(model_directory, 'model.h5'))

result = model.evaluate(test_x, test_t, verbose=0)
print(f'\n(Test Loss, Test Accuracy)=({result[0]}, {result[1]})')



