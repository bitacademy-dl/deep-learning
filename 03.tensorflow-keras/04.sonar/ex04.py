# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Model Fitting #3 - K-Fold Cross Validation
import os

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# 1-2. 10-fold Cross Validation
nfold = 10
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
accuracies = []

for mask_train, mask_test in skf.split(x, t):
    # 2. model frame config
    model = Sequential()
    model.add(Dense(20, input_dim=60, activation='relu'))
    model.add(Dense(10, input_dim=20, activation='relu'))
    model.add(Dense(2, input_dim=10, activation='softmax'))

    # 3. model fitting config
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. model fitting
    model.fit(x[mask_train], tf.keras.utils.to_categorical(t[mask_train]), epochs=200, batch_size=5, verbose=1)

    # 5. result
    result = model.evaluate(x[mask_test], tf.keras.utils.to_categorical(t[mask_test]), verbose=0)
    accuracies.append(result[1])


print(f'\n{nfold} fold accuracies:{accuracies}')





