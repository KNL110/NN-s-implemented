# multi-layered network of perceptrons with feed-forward architecture
import numpy as np
from GradOptimizers import ActivationFunction, ActivationGradient

LossFxn = {
    'MSE': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    'Entropy': lambda y_true, y_pred: np.sum(y_true * np.log(1/y_pred))
}

LossFxnGrad = {
    'MSE': lambda y_true, y_pred: (y_pred - y_true) / y_true.size,
    'Entropy': lambda y_true, y_pred: (-y_true / (y_pred))
}


class Dense:
    def __init__(self, input_size, output_size, activation=None):
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.Z = self.W @ input + self.b

        if self.activation is None:
            return self.Z
        self.output = ActivationFunction[self.activation](self.Z)
        return self.output

    def backward(self, grad_output, lr):
        #Gradient w.r.t. hidden units
        if self.activation is not None:

class NeuralNetwork:
    def __init__(self, layerSizes, activations, LossFunction='MSE'):
        self.L = len(activations)
        self.layers = {}
        for i in range(self.L):
            activation = None if activations[i] is None else activations[i]
            self.layers[i+1] = Dense(layerSizes[i], layerSizes[i + 1], activation)
        self.LossFunction = LossFunction

    def forward(self, x):
        a = x
        for layer in self.layers.values():
            a = layer.forward(a)
        return a
    
    def backward(self,y_true, y_pred):
        #compute output layer gradient
        grad_al = -(y_true - y_pred)
        for l in range(self.L-1, 0, -1):
            layer = self.layers[l]
            Wl = self.layers[l+1].W
            bl = self.layers[l+1].b
            grad_hl = Wl.T @ grad_al
            grad_al = grad_hl * ActivationGradient[layer.activation](layer.Z)
            

    def compute_loss(self, y_true, y_pred):
        return LossFxn[self.LossFunction](y_true, y_pred)

    def train(self, X, Y, lr=0.01, epochs=1000, batchSize=32):
        for epoch in range(epochs):
            totalLoss = 0
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)

                y_pred = self.forward(x)

                loss = self.compute_loss(y, y_pred)
                totalLoss += loss
                # Backward pass would go here
            print(f"Epoch {epoch}, Loss: {totalLoss}")


# Xor problem example
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(
        layerSizes=[2, 4, 1],
        activations=['sigmoid', 'sigmoid'],
        LossFunction='MSE'
        )

    # nn.train(X, Y, lr=0.5, epochs=1000)

    # for x in X:
    #     y_pred = nn.forward(x.reshape(-1, 1))
    #     print(f"Input: {x}, Predicted Output: {y_pred.flatten()[0]:.4f}")
