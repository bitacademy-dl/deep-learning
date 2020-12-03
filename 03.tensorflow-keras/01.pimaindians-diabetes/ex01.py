# Pima Indians Diabetes Prediction Model(피마 인디언 당요병 예측 모델)
# Explore Dataset(데이터 탐색)
import pandas as pd

df = pd.read_csv('./dataset/pimaindians-diabetes.csv', names=['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'predigree', 'age', 'class'])

# 처음 5줄을 읽는다.
print(df.head(5))

# 데이터의 전반적인 정보
print(df.info())

# thickness 와 class 만을 출력하고 싶을 때
print(df[['thickness', 'class']])
