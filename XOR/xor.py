from dense import Dense
from utils import TanH, Sigmoid, Relu
from losses import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
    TanH()
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