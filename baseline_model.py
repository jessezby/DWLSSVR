
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class LSSVMRegression(BaseEstimator, RegressorMixin):

    def __init__(self,
                 gamma: float = 1.0,
                 kernel: str = None,
                 c: float = 1.0,
                 d: float = 2,
                 sigma: float = 1.0):
        self.gamma = gamma
        self.c = c
        self.d = d
        self.sigma = sigma
        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel

        params = dict()
        if kernel == 'poly':
            params['c'] = c
            params['d'] = d
        elif kernel == 'rbf':
            params['sigma'] = sigma

        self.kernel_ = LSSVMRegression.__set_kernel(self.kernel, **params)

        #model parameters
        self.x = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None

    def get_params(self, deep=True):

        return {
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "sigma": self.sigma
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        params = dict()
        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d
        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma
        self.kernel_ = LSSVMRegression.__set_kernel(self.kernel, **params)

        return self

    def set_attributes(self, **parameters):
        for param, value in parameters.items():
            if param == 'intercept_':
                self.intercept_ = value
            elif param == 'coef_':
                self.coef_ = value
            elif param == 'support_':
                self.x = value

    @staticmethod
    def __set_kernel(name: str, **params):

        def linear(xi, xj):
            """
               v*v=scal (dot-product OK)
               v*m=v    (dot-product OK)
               m*m=m    (matmul for 2Dx2D, ok with dot-product)
            """
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            return ((np.dot(xi, xj.T)) / c + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            from scipy.spatial.distance import cdist

            if (xi.ndim == 2 and xi.ndim == xj.ndim):  
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) /
                              (2 * (sigma**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape) - 1  
                return np.exp(-(np.dot(xi, xi) +
                                (xj**2).sum(axis=ax) - 2 * np.dot(xi, xj.T)) /
                              (2 * (sigma**2)))
            else:
                message = "The rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else:  
            message = "Kernel " + name + " is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)

    def __OptimizeParams(self):

        Omega = self.kernel_(self.x, self.x)
        Ones = np.array(
            [[1]] *
            len(self.y))  

        A_dag = np.linalg.pinv(
            np.block([[0, Ones.T],
                      [
                          Ones,
                          Omega + self.gamma**-1 * np.identity(len(self.y))
                      ]]))  
        B = np.concatenate((np.array([0]), self.y), axis=None)

        solution = np.dot(A_dag, B)
        self.intercept_ = solution[0]
        self.coef_ = solution[1:]

    def fit(self, X: np.ndarray, y: np.ndarray):

        if isinstance(
                X, (pd.DataFrame,
                    pd.Series)):  
            Xloc = X.to_numpy()
        else:
            Xloc = X

        if isinstance(y, (pd.DataFrame, pd.Series)):
            yloc = y.to_numpy()
        else:
            yloc = y
        if (Xloc.ndim == 2) and (yloc.ndim == 1 or yloc.ndim == 2):
            self.x = Xloc
            self.y = yloc
            self.__OptimizeParams()
        else:
            message = "The fit procedure requires a 2D numpy array of features "\
                "and 1D array of targets"
            raise Exception(message)

    def predict(self, X: np.ndarray) -> np.ndarray:

        Ker = self.kernel_(
            X,
            self.x)  
        Y = np.dot(self.coef_, Ker.T) + self.intercept_
        return Y
