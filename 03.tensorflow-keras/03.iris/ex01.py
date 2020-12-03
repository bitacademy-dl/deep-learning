# Iris Species Multi-Class Classification Model(Iris 품종 예측 모델)
# Explore Dataset(데이터 탐색)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

dateset_file = './dataset/iris.csv'
df = pd.read_csv(dateset_file, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
print(df.info())
print(df.head())

# 데이터 분류
dataset = df.values
x, t = dataset[:, 0:4], dataset[:, 4]

# print(x.shape)
# print(t.shape)

# 문자열을 숫자(One-Hot) 바꾸기
# [1 0 0] = 'Iris-setosa'
# [0 1 0] = 'Iris-versicolor'
# [0 0 1] = 'Iris-virginica'

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)
print(t)

