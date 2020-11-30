# Myltiply Layer Test
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Multiply
except ImportError:
    print('Library Module Can Not Found')

# data
apple = 100
count = 5
discount = 0.9

# layers
layer1 = Multiply()
layer2 = Multiply()


# forward
appleprice = layer1.forward(apple, count)
print(f'appleprice={appleprice}')
totalprice = layer2.forward(appleprice, discount)
print(f'totalprice={totalprice}')

print("=====================================")

# backward propagation
dappleprice, ddiscount = layer2.backward(1)
print(f'dappleprice={dappleprice}, dcount={ddiscount}')

dapple, dcount = layer1.backward(dappleprice)
print(f'dapple={dapple}, dcount={dcount}')


