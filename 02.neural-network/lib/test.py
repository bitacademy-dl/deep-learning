import numpy as np

print('= nditrator test =====================')
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

print('= mask test =====================')

x = np.array([10, -10, 2, 0, -2])
mask = (x <= 0)
print(mask)

# forward
out = x.copy()
print(out)
out[mask] = 0
print(out)

# backward
dout = np.array([-1, 10, 2, 0, -2])
print(dout)
dout[mask] = 0
print(dout)
