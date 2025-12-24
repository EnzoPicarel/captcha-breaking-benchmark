import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.stats = {}
        self.priors = {}
        self.classes = None

    def _gaussian_log_density(self, x, mean, var):
        # log-density formula
        eps = 1e-9
        coeff = -0.5 * np.log(2 * np.pi * var + eps)
        exponent = -0.5 * ((x - mean) ** 2) / (var + eps)
        return np.sum(coeff + exponent)

    def fit(self, X, y):
        n_samples = len(X)
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y.flatten() == c]
            
            # priors
            self.priors[c] = len(X_c) / n_samples
            
            # mean/var per feature
            self.stats[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0)
            }

    def predict(self, X):
        y_pred = []
        total = len(X)
        for i, x in enumerate(X):
            if i % 5000 == 0:
                print(f"      predicting {i}/{total}", end='\r')
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = self._gaussian_log_density(
                    x, self.stats[c]['mean'], self.stats[c]['var']
                )
                posteriors.append(prior + likelihood)
            
            # map estimate
            y_pred.append(self.classes[np.argmax(posteriors)])
        print(f"      predicting {total}/{total}", end='\r')
        return np.array(y_pred).reshape(-1, 1)