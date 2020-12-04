# Housing Price Prediction(Linear Regression) Model(보스톤 집값 예측 모델)
# Model fitting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1.load training/test data
dateset_file = './dataset/housing.csv'
df = pd.read_csv(dateset_file, delim_whitespace=True, header=None)

dataset = df.values
x = dataset[:, 0:13]
t = dataset[:, 13]

# print(x.shape)
# print(t.shape)
train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. model fitting config
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. model fitting
model.fit(train_x, train_t, epochs=200, batch_size=10)

# 5. prediction test
y = model.predict(test_x).flatten()
for i in range(10):
    label = test_t[i]
    prediction = y[i]
    print(f'실제가격:{label:.3f}, 예상가격:{prediction:.3f}')



