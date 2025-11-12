# multi-layered network of perceptrons with feed-forward architecture
import numpy as np
from GradOptimizers import ActivationFunction, ActivationGradient

LossFxn = {
    'MSE': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    'CrossEntropy': lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
}

LossFxnGrad = {
    'MSE': lambda y_true, y_pred: (y_pred - y_true) / y_true.size,
    'CrossEntropy': lambda y_true, y_pred: -(y_true / (y_pred + 1e-9)) + (1 - y_true) / (1 - y_pred + 1e-9)
}

class Neuron:
    def __init__(self, d, activation):
        self.weights = np.random.randn(d, 1) * np.sqrt(2. / d)
        self.bias = np.zeros(1)
        self.activation = activation
        self.activate = ActivationFunction[activation]

    def forward(self, x):
        self.input = x
        self.z = x.T @ self.weights + self.bias
        self.output = self.activate(self.z)
        return self.output

    def backward(self, dout):
        pass  # Placeholder for backward method


class NeuralLayer:
    def __init__(self, n_neurons, d_input, activation):
        self.activation = activation
        self.neurons = [Neuron(d_input, activation) for _ in range(n_neurons)]
        self.weights = np.hstack([neuron.weights for neuron in self.neurons])
        self.biases = np.array([neuron.bias for neuron in self.neurons]).reshape(-1, 1)

    def forward(self, x):
        self.input = x
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(x))
        self.z = np.hstack([neuron.z for neuron in self.neurons])
        self.output = np.vstack(outputs)
        return self.output

    def backward(self, weights, dout):
        self.grad_hi = weights.T @ dout
        self.grad_ai = self.grad_hi * ActivationGradient[self.activation](self.z)
        self.grad_w = self.grad_ai @ self.input.T
        self.grad_b = self.grad_ai


class NeuralNetwork:
    def __init__(self, layerSizes, activations, LossFunction='CrossEntropy', optimizer=None):
        self.n_layers = len(layerSizes)
        self.HiddenLayers = self.n_layers - 2
        self.layers = {}
        self.optimizer = optimizer
        self.LossFunction = LossFunction
        self.GradLossFxn = None
        for i in range(1, self.n_layers):
            self.layers[i] = NeuralLayer(layerSizes[i], layerSizes[i - 1], activations[i-1])

    def grad_w(self, x, y):
        return np.zeros_like(x)

    def forward(self, x):
        self.input = x
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    
    def Backward(self, y_true, y_pred):
        dL_da = LossFxnGrad[self.LossFunction](y_true, y_pred)
        
        for idx in range(self.n_layers - 1, 0, -1):
            layer = self.layers[idx]
            
            if idx == self.n_layers - 1:
                dL_dz = dL_da * ActivationGradient[layer.activation](layer.z.reshape(-1, 1))
            else:
                next_layer = self.layers[idx + 1]
                dL_da = next_layer.weights @ next_layer.grad_ai
                dL_dz = dL_da * ActivationGradient[layer.activation](layer.z.reshape(-1, 1))
            
            layer.grad_ai = dL_dz
            layer.grad_w = dL_dz @ layer.input.T
            layer.grad_b = dL_dz


    def train(self,model, x, y, lr=0.01, epochs=1000, verbose=True):
        n,d = x.shape
        for epoch in range(epochs):
            total_loss = 0
            for i in range(n):
                y_pred = self.forward(x[i].reshape(-1,1))
                self.Backward(y[i].reshape(-1,1), y_pred)
                
                for layer in self.layers.values():
                    layer.weights -= lr * layer.grad_w.T
                    layer.biases -= lr * layer.grad_b
                    for j, neuron in enumerate(layer.neurons):
                        neuron.weights = layer.weights[:, j:j+1]
                        neuron.bias = layer.biases[j:j+1]
                loss = LossFxn[self.LossFunction](y[i].reshape(-1,1), y_pred)
                total_loss += loss
            if verbose:
                print(f"Epoch {epoch}: Loss = {total_loss / n:.6f}")



#Xor problem example
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(
        layerSizes=[2, 5, 1],
        activations=['tanh', 'sigmoid'],
        LossFunction='MSE'
        )

    nn.train(nn, X, Y, lr=0.5, epochs=1000)

    print("\nXOR Results:")
    for x in X:
        y_pred = nn.forward(x.reshape(-1, 1))
        print(f"Input: {x}, Predicted: {y_pred.flatten()[0]:.4f}")