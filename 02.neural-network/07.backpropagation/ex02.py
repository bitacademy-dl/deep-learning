# Myltiply & Add Layer Test
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Multiply, Add
except ImportError:
    print('Library Module Can Not Found')

# data
apple = 100
applecount = 3
orange = 200
orangecount = 5
discount = 0.9

# layers

# forward
appleprice = 0
orangeprice = 0
appleorangeprice = 0
totalprice = 0

print("=====================================")

# backward propagation
dtotalprice = 1

dappleorangeprice = 0
ddiscount = 0

dappleprice = 0
dorangeprice = 0

dapple = 0
dapplecount = 0

dorange = 0
dorangecount = 0