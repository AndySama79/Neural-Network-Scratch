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
    W1 = np.random.randn(n_h, n_x - 1) * 0.01
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
    return np.exp(Z) / np.sum(np.exp(Z)), Z

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
    m = Y.shape[1]

    cost = -(1 / m) * (np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL.T)))

    cost = np.squeeze(cost)

    return cost

def d_relu(dA, activation_cache):
    Z = activation_cache
    return dA * (Z > 0)

def d_sigmoid(dA, activation_cache):
    Z = activation_cache
    return dA * (Z(1 - Z))

def d_softmax(dA, activation_cache, Y=None):
    Z = activation_cache
    A, _ = softmax(Z)
    return dA * (A - Y)

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, Y=None):
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = d_sigmoid(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = d_sigmoid(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = d_softmax(dA, activation_cache, Y)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2    # since pair of two (W1, b1), (W2, b2),...

    for l in range(L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
    
    return parameters


# %%    Building an L-layer DNN
def initialize_parameters_deep(layer_dims: list) -> dict:
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2    # since pairs

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)

    return AL, caches

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    cache = caches[-1]  #   last indexed cache
    dA_prev, dW, db = linear_activation_backward(dAL, cache, "softmax", Y)  # have to pass Y for softmax derivative
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(L-1)):
        cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l+1)], cache, "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db

    return grads
# %% creating and training the models
