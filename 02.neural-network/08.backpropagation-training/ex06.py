# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: MultiLayerNet
# Estimation: Training
import os
import pickle
from matplotlib import pyplot as plt

train_loss_file = os.path.join(os.getcwd(), 'model', 'twolayer_trainloss.pkl')
train_losses = None

with open(train_loss_file, 'rb') as f:
    train_losses = pickle.load(f)

plt.plot(train_losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.show()



