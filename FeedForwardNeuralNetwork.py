# multi-layered network of perceptrons with feed-forward architecture
import numpy as np
from GradOptimizers import ActivationFunction, ActivationGradient, LossFxn, LossFxnGrad, SGD, RMSProp, Adam, Nadam


class Dense:
    def __init__(self, input_size, output_size, activation=None):
        limit = np.sqrt(6 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.activation = activation
        self.dW = None
        self.db = None

    def forward(self, input):
        self.input = input
        self.Z = self.W @ input + self.b

        if self.activation is None:
            return self.Z
        self.output = ActivationFunction[self.activation](self.Z)
        return self.output

    def backward(self, grad_output):
        # Gradient w.r.t. pre-activation Z
        if self.activation is None:
            grad_Z = grad_output
        else:
            grad_Z = grad_output * ActivationGradient[self.activation](self.Z)

        # Gradients w.r.t. weights and biases (summed over batch)
        m = grad_output.shape[1]  # Batch size
        self.dW = (grad_Z @ self.input.T) / m
        self.db = np.sum(grad_Z, axis=1, keepdims=True) / m

        # Gradient w.r.t. input (to pass to previous layer)
        grad_input = self.W.T @ grad_Z

        return grad_input


class NeuralNetwork:
    def __init__(self, layerSizes, activations, LossFunction='MSE', optimizer='SGD', optimizer_params={}):
        self.L = len(activations)
        self.layers = {}
        for i in range(self.L):
            activation = None if activations[i] is None else activations[i]
            self.layers[i+1] = Dense(layerSizes[i],
                                     layerSizes[i + 1], activation)
        self.LossFunction = LossFunction

        # Initialize optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD(**optimizer_params)
        elif optimizer == 'RMSProp':
            self.optimizer = RMSProp(**optimizer_params)
        elif optimizer == 'Adam':
            self.optimizer = Adam(**optimizer_params)
        elif optimizer == 'Nadam':
            self.optimizer = Nadam(**optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def forward(self, x):
        a = x
        for layer in self.layers.values():
            a = layer.forward(a)
        return a

    def backward(self, y_true, y_pred):
        # Compute gradient of loss w.r.t. output
        grad = LossFxnGrad[self.LossFunction](y_true, y_pred)

        # Backpropagate through layers
        for i in range(self.L, 0, -1):
            grad = self.layers[i].backward(grad)

    def compute_loss(self, y_true, y_pred):
        return LossFxn[self.LossFunction](y_true, y_pred)

    def train(self, X, Y, epochs=1000, batchSize=32, verbose=True):
        m = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            totalLoss = 0
            num_batches = 0

            for i in range(0, m, batchSize):
                # Transpose for (features, batch_size)
                x_batch = X_shuffled[i:i+batchSize].T
                y_batch = Y_shuffled[i:i+batchSize].T

                # Forward pass
                y_pred = self.forward(x_batch)

                # Compute loss
                loss = self.compute_loss(y_batch, y_pred)
                totalLoss += loss
                num_batches += 1

                # Backward pass
                self.backward(y_batch, y_pred)

                # Optimizer update
                self.optimizer.update(self.layers)

            if verbose:
                print(f"Epoch {epoch}, Loss: {totalLoss / num_batches}")


# Xor problem example
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])


    print("Training with Adam and MSE...")
    nn = NeuralNetwork(
            layerSizes=[2, 5, 1],
            activations=['sigmoid', 'sigmoid'],
            LossFunction='MSE',
            optimizer='Adam',
            optimizer_params={'lr': 0.01}
        )
    nn.train(X, Y, epochs=10000, batchSize=4, verbose=False)
    print("Final Loss:", nn.compute_loss(Y.T, nn.forward(X.T)))
    print("\nPredictions:")
    for x in X:
        y_pred = nn.forward(x.reshape(-1, 1))
        if y_pred >= 0.5:
            pred_class = 1
        else:
            pred_class = 0
        print(f"Input: {x}, Predicted Output: {y_pred.ravel()[0]:.4f}, Predicted Class: {pred_class}")
