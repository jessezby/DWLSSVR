"""
LSSVMRegression: Least Squares Support Vector Machine for Regression

This module implements a Least Squares Support Vector Machine (LS-SVM) for regression tasks.
It is built on the BaseEstimator and RegressorMixin base classes from scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.spatial.distance import cdist

class LSSVMRegression(BaseEstimator, RegressorMixin):
    """
    Least Squared Support Vector Machine (LS-SVM) regression class.

    Attributes:
        gamma (float): Regularization parameter.
        kernel (str): Kernel type ('linear', 'poly', or 'rbf').
        c (float): Scaling constant for polynomial kernel.
        d (float): Polynomial degree for polynomial kernel.
        sigma (float): Scaling constant for RBF kernel.
        x (np.ndarray): Training data features.
        y (np.ndarray): Training data targets.
        coef_ (np.ndarray): Coefficients of the support vectors.
        intercept_ (float): Intercept term.
    """

    def __init__(self, gamma=1.0, kernel=None, c=1.0, d=2, sigma=1.0):
        """
        Initialize the LSSVMRegression.

        Args:
            gamma (float): Regularization parameter.
            kernel (str): Kernel type ('linear', 'poly', or 'rbf').
            c (float): Scaling constant for polynomial kernel.
            d (float): Polynomial degree for polynomial kernel.
            sigma (float): Scaling constant for RBF kernel.
        """
        self.gamma = gamma
        self.c = c
        self.d = d
        self.sigma = sigma
        self.kernel = kernel or 'rbf'

        params = {}
        if self.kernel == 'poly':
            params = {'c': c, 'd': d}
        elif self.kernel == 'rbf':
            params = {'sigma': sigma}

        self.kernel_ = self.__set_kernel(self.kernel, **params)
        self.x = self.y = self.coef_ = self.intercept_ = None

    @staticmethod
    def __set_kernel(name, **params):
        """
        Set the kernel function based on the given name and parameters.

        Args:
            name (str): Name of the kernel function.
            **params: Additional parameters for the kernel function.

        Returns:
            function: The kernel function.

        Raises:
            KeyError: If an unknown kernel is specified.
        """
        def linear(xi, xj):
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            return ((np.dot(xi, xj.T)) / c + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            if xi.ndim == 2 and xi.ndim == xj.ndim:
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) / (2 * (sigma**2)))
            elif xi.ndim < 2 and xj.ndim < 3:
                ax = len(xj.shape) - 1
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax) - 2 * np.dot(xi, xj.T)) / (2 * (sigma**2)))
            else:
                raise ValueError("The RBF kernel is not suited for arrays with rank > 2")

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
        if name in kernels:
            return kernels[name]
        else:
            raise KeyError(f"Kernel {name} is not implemented. Please choose from: {', '.join(kernels.keys())}")

    def __OptimizeParams(self):
        """Optimize the LS-SVM parameters."""
        Omega = self.kernel_(self.x, self.x)
        Ones = np.ones((len(self.y), 1))

        A_dag = np.linalg.pinv(np.block([
            [0, Ones.T],
            [Ones, Omega + self.gamma**-1 * np.eye(len(self.y))]
        ]))
        B = np.concatenate((np.array([0]), self.y), axis=None)

        solution = np.dot(A_dag, B)
        self.intercept_ = solution[0]
        self.coef_ = solution[1:]

    def fit(self, X, y):
        """
        Fit the LS-SVM model.

        Args:
            X (np.ndarray or pd.DataFrame): Training data features.
            y (np.ndarray or pd.Series): Training data targets.

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        X = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        if X.ndim == 2 and (y.ndim == 1 or y.ndim == 2):
            self.x = X
            self.y = y.reshape(-1, 1) if y.ndim == 1 else y
            self.__OptimizeParams()
        else:
            raise ValueError("The fit procedure requires a 2D array of features and a 1D or 2D array of targets")

    def predict(self, X):
        """
        Predict using the LS-SVM model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        Ker = self.kernel_(X, self.x)
        return np.dot(self.coef_, Ker.T) + self.intercept_

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "sigma": self.sigma
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        params = {}
        if self.kernel == 'poly':
            params = {'c': self.c, 'd': self.d}
        elif self.kernel == 'rbf':
            params = {'sigma': self.sigma}
        self.kernel_ = self.__set_kernel(self.kernel, **params)

        return self

    def set_attributes(self, **parameters):
        """
        Manually set the attributes of the model.

        This method should generally not be used, except for testing or creating an averaged model.

        Args:
            **parameters: Dictionary of parameters to set.
                - 'intercept_': float intercept
                - 'coef_': array of coefficients
                - 'support_': array of support vectors
        """
        for param, value in parameters.items():
            if param == 'intercept_':
                self.intercept_ = value
            elif param == 'coef_':
                self.coef_ = value
            elif param == 'support_':
                self.x = value