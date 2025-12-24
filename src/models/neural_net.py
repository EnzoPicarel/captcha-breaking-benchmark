import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=16, learning_rate=0.1, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.params = {}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        m = X.shape[0]
        np.random.seed(42)
        
        # he init
        scale = np.sqrt(2 / self.input_size)
        self.params = {
            'W1': np.random.randn(self.input_size, self.hidden_size) * scale,
            'b1': np.zeros((1, self.hidden_size)),
            'W2': np.random.randn(self.hidden_size, 1) * scale,
            'b2': np.zeros((1, 1))
        }

        for _ in range(self.epochs):
            # forward
            Z1 = np.dot(X, self.params['W1']) + self.params['b1']
            A1 = self._sigmoid(Z1)
            Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
            A2 = self._sigmoid(Z2)
            
            # backward
            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            
            dZ1 = np.dot(dZ2, self.params['W2'].T) * self._sigmoid_derivative(A1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            
            # update
            self.params['W1'] -= self.lr * dW1
            self.params['b1'] -= self.lr * db1
            self.params['W2'] -= self.lr * dW2
            self.params['b2'] -= self.lr * db2

    def predict(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self._sigmoid(Z2)
        return (A2 >= 0.5).astype(int)