
# Scientific and vector computation for python
import numpy as np



class LogisticRegressionUtils:

    @staticmethod
    def sigmoid(z):
        # sigmoid calculation 
        g = (1 / (1 + np.exp(-z)))
        return g
    
    @staticmethod
    def costFunction(w, X, y):
        m = y.size  # number of training examples
        
        # Add intercept term to X
        X = np.concatenate([np.ones((m, 1)), X], axis=1)

        J = 0
        grad = np.zeros(w.shape)

        h = LogisticRegressionUtils.sigmoid(X @ w)
        term1 = y * np.log(h)
        term2 = (1 - y) * np.log(1 - h)
        J = -(1 / m) * np.sum(term1 + term2)
        
        error = h - y 
        grad = (1 / m) * (X.T @ error) 
        return J, grad