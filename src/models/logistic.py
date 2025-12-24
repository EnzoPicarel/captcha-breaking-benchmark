import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None

    def _sigmoid(self, z):
        # clip for stability
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        m, n = X.shape
        # bias trick
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros((n + 1, 1))
        
        for i in range(self.n_iters):
            if i % 1000 == 0:
                print(f"      iteration {i}/{self.n_iters}", end='\r')
            z = np.dot(X_b, self.theta)
            h = self._sigmoid(z)
            
            # gradient ascent
            gradient = np.dot(X_b.T, (y - h)) 
            self.theta += self.lr * (gradient / m)
        print(f"      iteration {self.n_iters}/{self.n_iters}", end='\r')

    def predict(self, X):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        prob = self._sigmoid(np.dot(X_b, self.theta))
        return (prob >= 0.5).astype(int)