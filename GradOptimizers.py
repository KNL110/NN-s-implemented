import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = np.array(x).flatten()
    x_max = np.max(x)
    x_shifted = np.clip(x - x_max, -500, 500)
    exp_x = np.exp(x_shifted)
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
    'softmax': lambda x: np.ones_like(x)
}

def grad_w(x, y):
    return np.zeros_like(x)