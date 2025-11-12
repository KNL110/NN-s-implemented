import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = np.array(x).reshape(-1)
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)

ActivationFunction = {
    'sigmoid': sigmoid,
    'relu': lambda x: np.maximum(0, x),
    'softmax': softmax,
    'tanh': lambda x: np.tanh(x)
}

ActivationGradient = {
    'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
    'relu': lambda x: np.where(x > 0, 1, 0),
    'tanh': lambda x: 1 - np.tanh(x)**2,
    'softmax': lambda x: 1
}

def grad_w(x, y):
    return np.zeros_like(x)