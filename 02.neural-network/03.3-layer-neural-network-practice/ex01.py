# 3층 신경망 신호 전달 구현1: 은닉 1층 전달
import numpy as np

print('\n= 신호 전달 구현1: 은닉 1층 전달 ======================')
x = np.array([1., 5.])
print(f'x dimension: {x.shape}')        # 2 vector

w1 = np.array([
    [0.1, 0.2, 0.5],
    [0.3, 0.4, 1.]
])
print(f'w1 dimension: {w1.shape}')      # 2 x 3 matrix

b1 = np.array([0.1, 0.2, 0.3])
print(f'b1 dimension: {b1.shape}')      # 3 vector

# 오류: 일차함수(식) 중심으로 생각하지 말고 신호(데이터, 값) 중심으로 생각
# a1 = np.dot(w1, x) + b1
a1 = np.dot(x, w1) + b1
print(f'a1 = {a1}')

# tensor flow~
# 2(v) -> @ 2x3(m) -> 3(v) -> + 3(v) -> 3(v)
# tensor1(크기2) 입력 신호가 뉴런에서 tensor2(가중치)와 총합으로 tensor1(크기3)가 출력신호가 되었다.

