# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Model Fitting #1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1.load training/test data
dateset_file = './dataset/sonar.csv'
df = pd.read_csv(dateset_file, header=None)

dataset = df.values
x, t = dataset[:, 0:60].astype(float), dataset[:, 60]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)

# 2. model frame config
model = Sequential()
model.add(Dense(20, input_dim=60, activation='relu'))
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(2, input_dim=10, activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=200, batch_size=5, verbose=1)

# 5. result
loss = history.history['loss']
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy)=({result[0]}, {result[1]})')


