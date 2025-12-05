import numpy as np

# --- Activations ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def identity(x):
    return x

ActivationFunction = {
    'sigmoid': sigmoid,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softmax': softmax,
    'tanh': tanh,
    'identity': identity
}

ActivationGradient = {
    'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
    'relu': lambda x: np.where(x > 0, 1, 0),
    'leaky_relu': lambda x: np.where(x > 0, 1, 0.01),
    'tanh': lambda x: 1 - np.tanh(x)**2,
    'softmax': lambda x: np.ones_like(x), # Jacobian is complex, handled in loss usually, but for simple backprop we often assume simplified gradient or handle with CrossEntropy
    'identity': lambda x: np.ones_like(x)
}

# --- Loss Functions ---

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_grad(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[1]  # Divide by batch size (axis 1)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_grad(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size

LossFxn = {
    'MSE': mse,
    'CrossEntropy': cross_entropy,
    'BinaryCrossEntropy': binary_cross_entropy
}

LossFxnGrad = {
    'MSE': mse_grad,
    'CrossEntropy': cross_entropy_grad,
    'BinaryCrossEntropy': binary_cross_entropy_grad
}

# --- Optimizers ---

class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def update(self, layers):
        for i, layer in layers.items():
            if not hasattr(layer, 'W'): continue
            
            if i not in self.velocities:
                self.velocities[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }
            
            v = self.velocities[i]
            
            # Update velocities
            v['W'] = self.momentum * v['W'] + self.lr * layer.dW
            v['b'] = self.momentum * v['b'] + self.lr * layer.db
            
            if self.nesterov:
                layer.W -= (self.momentum * v['W'] + self.lr * layer.dW)
                layer.b -= (self.momentum * v['b'] + self.lr * layer.db)
            else:
                layer.W -= v['W']
                layer.b -= v['b']

class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layers):
        for i, layer in layers.items():
            if not hasattr(layer, 'W'): continue

            if i not in self.cache:
                self.cache[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }
            
            s = self.cache[i]
            
            s['W'] = self.beta * s['W'] + (1 - self.beta) * (layer.dW ** 2)
            s['b'] = self.beta * s['b'] + (1 - self.beta) * (layer.db ** 2)
            
            layer.W -= self.lr * layer.dW / (np.sqrt(s['W']) + self.epsilon)
            layer.b -= self.lr * layer.db / (np.sqrt(s['b']) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for i, layer in layers.items():
            if not hasattr(layer, 'W'): continue

            if i not in self.m:
                self.m[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            
            m = self.m[i]
            v = self.v[i]
            
            # Update moments
            m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * layer.dW
            m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * layer.db
            
            v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * (layer.dW ** 2)
            v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * (layer.db ** 2)
            
            # Bias correction
            m_hat_W = m['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = m['b'] / (1 - self.beta1 ** self.t)
            
            v_hat_W = v['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = v['b'] / (1 - self.beta2 ** self.t)
            
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for i, layer in layers.items():
            if not hasattr(layer, 'W'): continue

            if i not in self.m:
                self.m[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            
            m = self.m[i]
            v = self.v[i]
            
            # Update moments
            m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * layer.dW
            m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * layer.db
            
            v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * (layer.dW ** 2)
            v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * (layer.db ** 2)
            
            # Bias correction
            m_hat_W = m['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = m['b'] / (1 - self.beta1 ** self.t)
            
            v_hat_W = v['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = v['b'] / (1 - self.beta2 ** self.t)
            
            # Nesterov update
            m_nesterov_W = self.beta1 * m_hat_W + (1 - self.beta1) * layer.dW / (1 - self.beta1 ** self.t)
            m_nesterov_b = self.beta1 * m_hat_b + (1 - self.beta1) * layer.db / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * m_nesterov_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.b -= self.lr * m_nesterov_b / (np.sqrt(v_hat_b) + self.epsilon)
