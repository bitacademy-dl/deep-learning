# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Explore Dataset(데이터 탐색)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

dateset_file = './dataset/sonar.csv'
df = pd.read_csv(dateset_file, header=None)
print(df.info())
print(df.head())
