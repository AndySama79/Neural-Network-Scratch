#%% imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%    load data
data_path = "data/digit-recognizer"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
# %%    explore
train_data = np.array(train_data)
np.random.shuffle(train_data)   # shuffle before splitting
m, n_x = train_data.shape
# %%    Splitting the train_data
train = train_data[0:30000].T
Y_train = train[0]
X_train = train[1:n_x]
X_train = X_train / 255 #   

test = train_data[30000:m].T
Y_test = test[0]
X_test = test[1:n_x]
X_test = X_test / 255

print("X_train:", X_train)
print("Y_train:", Y_train)
# %%    Building a two-layer NN (activation->ReLU->activation->softmax)
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def relu(Z: np.ndarray) -> np.ndarray:
    return np.max(0, Z), Z

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z)), Z

def softmax(Z: np.ndarray) -> np.ndarray:
    return Z / np.sum(np.exp(Z)), Z

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b, activation)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b, activation)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b, activation)
        A, activation_cache = softmax(Z)
    else:
        print(f"{activation} function not defined")

    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(AL, Y):
    pass

def linear_activation_backward(dA, cache, activation):
    pass

def update_parameters(parameters, grads, learning_rate):
    pass


# %%    Building an L-layer DNN
def initialize_parameters_deep(layer_dims: list) -> dict:
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def L_model_forward(X, parameters):
    
