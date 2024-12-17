# Author: Malte Bengtsson, Karl Eriksson
# Scientific and vector computation for python
import numpy as np



class LogisticRegressionUtils:

    @staticmethod
    def sigmoid(z):
        # sigmoid calculation 
        g = (1 / (1 + np.exp(-z)))
        return g
    
    @staticmethod
    def costFunction(w, X, y, lambda_):
        m = y.size  # number of training examples

        # add intercept term to X
        X = np.concatenate([np.ones((m, 1)), X], axis=1)
        

        # compute hypothesis
        #h = LogisticRegressionUtils.sigmoid(X @ w)
        h = np.clip(LogisticRegressionUtils.sigmoid(X @ w), 1e-10, 1 - 1e-10)


        # cost function without regularization
        term1 = y * np.log(h)
        term2 = (1 - y) * np.log(1 - h)
        J = -(1 / m) * np.sum(term1 + term2)

        # Add regularization term to the cost (excluding the bias term)
        reg_term = (lambda_ / (2 * m)) * np.sum(w[1:] ** 2)
        J += reg_term

        # compute gradient without regularization
        error = h - y
        grad = (1 / m) * (X.T @ error)

        # add regularization term to the gradient (excluding the bias term)
        reg_grad = (lambda_ / m) * w
        reg_grad[0] = 0 
        grad += reg_grad

        return J, grad
