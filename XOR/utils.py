from activation import Activation
import numpy as np

#*  Hyperbolic Tangent
class TanH(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.power(np.tanh(x), 2)
        super().__init__(tanh, tanh_prime)

#*  Sigmoid
class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) - np.power(sigmoid(x), 2)
        super().__init__(sigmoid, sigmoid_prime)

#*  ReLU
class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime)