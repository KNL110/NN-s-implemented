import numpy as np

ActivationFunction = {
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'relu': lambda x: np.maximum(0, x),
    'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
    'tanh': lambda x: np.tanh(x)
}

ActivationGradient = {
    'sigmoid': lambda x: ActivationFunction['sigmoid'](x) * (1 - ActivationFunction['sigmoid'](x)),
    'relu': lambda x: np.where(x > 0, 1, 0),
    'tanh': lambda x: 1 - np.tanh(x)**2,
    'softmax': lambda x: 1
}

def grad_w(x, y):
    return np.zeros_like(x)