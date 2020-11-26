import numpy as np

# x = np.random.randn(2, 3)
x = np.arange(0., 10., 1.)
print(x)

it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

# test1
# for i in it:
#     print(i)


# test2
while not it.finished:
    idx = it.multi_index
    print(idx, x[idx])

    it.iternext()