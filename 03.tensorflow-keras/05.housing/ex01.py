# Housing Price Prediction(Linear Regression) Model(보스톤 집값 예측 모델)
# Explore Dataset(데이터 탐색)

import pandas as pd

dateset_file = './dataset/housing.csv'
df = pd.read_csv(dateset_file, header=None)
# print(df.info())
print(df.head())
