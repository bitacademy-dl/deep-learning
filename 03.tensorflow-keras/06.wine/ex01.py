# Wine Binary Classification Model(와인 종류 분류 모델)
# Exploring Dataset
import pandas as pd
import numpy as np

dateset_file = './dataset/wine.csv'
df = pd.read_csv(dateset_file, header=None)
df = df.sample(frac=1)

# print(df.info())
# print(df.head())

# 데이터 분류
dataset = df.values
x = dataset[:, 0:12]
t = dataset[:, 12]

# One-Hot 만들기
t = t[:, np.newaxis]
t = np.c_[t, t == 0]

print(x.shape)
print(t.shape)




