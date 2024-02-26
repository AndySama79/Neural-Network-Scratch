from dense import Dense
from utils import TanH, Sigmoid, Relu
from losses import mse, mse_prime
import numpy as np
import pandas as pd

train = pd.read_csv("../mNIST_digit_recognizer/data/digit-recognizer/train.csv")

Y = train['label'].to_numpy()
X = train.drop(['label'], axis=1).to_numpy() / 255

network = [
    Dense(784, 10),
    Relu(),
    Dense(10, 1),
    Sigmoid()
]

epochs = 10000
learning_rate = 0.1

#*  Train
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        #   forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        #   error
        error += mse(y, output)

        #   backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
        
    error /= len(X)
    if (e+1) % 100 == 0:
        print('%d/%d, error=%f' % (e+1, epochs, error))