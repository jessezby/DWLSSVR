"""
KMMXY_AUTO_DWLSSVMRegression: A class for Domain Adaptation in Regression Tasks

This module implements a Least Squares Support Vector Machine (LS-SVM) regression
with automatic Kernel Mean Matching (KMM) for domain adaptation. It combines
feature-wise KMM with a weighted LS-SVM to handle domain adaptation between source
and target domains.

Key features:
- Automatic tuning of KMM parameters
- Support for various kernels (linear, polynomial, RBF)
- Integration with scikit-learn's BaseEstimator and RegressorMixin
- Options for different optimization methods in KMM tuning
"""

import numpy as np
import pandas as pd
import json
import math
from sklearn.base import BaseEstimator, RegressorMixin
from cvxopt import matrix, solvers
from scipy.stats import median_abs_deviation
from sklearn.svm import SVR
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels
from geomloss import SamplesLoss
import torch
from scipy import optimize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from importance_weighted_cross_validation import cross_val_scores_weighted
from sklearn.metrics import mean_squared_error
from layers import SinkhornDistance


class KMMXY_AUTO_DWLSSVMRegression(BaseEstimator, RegressorMixin):
    """
    An auto-tuning Double-Weighted Least Squares Support Vector Machine (DW-LSSVM)
    regression class with Kernel Mean Matching (KMM) for domain adaptation.

    This class combines feature-wise KMM with a weighted LS-SVM to handle
    domain adaptation between source and target domains in regression tasks.
    """

    def __init__(self, X_source, y_source, gamma=1.0, kernel=None, KMM_kernel=None,
                 c=1.0, d=2, sigma=1.0, eta=1.0, S=5, method=None):
        """
        Initialize the KMMXY_AUTO_DWLSSVMRegression model.

        Args:
            X_source (np.ndarray): Source domain feature matrix
            y_source (np.ndarray): Source domain target values
            gamma (float): Regularization parameter for LSSVM
            kernel (str): Kernel type for LSSVM ('linear', 'poly', 'rbf')
            KMM_kernel (str): Kernel type for KMM ('KMM_rbf')
            c (float): Scaling constant for polynomial kernel
            d (int): Polynomial degree for polynomial kernel
            sigma (float): RBF kernel parameter for LSSVM
            eta (float): RBF kernel parameter for KMM
            S (int): Number of iterations for weighted LSSVM
            method (str): Optimization method for KMM tuning
        """
        self.gamma = gamma
        self.c=c  
        self.d = d
        self.sigma = sigma  
        self.eta = eta  
        self.S = S  
        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel
        if KMM_kernel is None:
            self.KMM_kernel = 'KMM_rbf'
        else:
            self.KMM_kernel = KMM_kernel  
        if method is None:
            self.method = 'minimize_scalar'
        else:
            self.method = method

        params = dict()
        if kernel == 'poly':
            params['c'] = c
            params['d'] = d
        elif kernel == 'rbf':
            params['sigma'] = sigma
        if KMM_kernel == 'KMM_rbf':
            params['eta'] = eta

        self.kernel_ = KMMXY_AUTO_DWLSSVMRegression.__set_kernel(
            self.kernel, **params)
        self.KMM_kernel_ = KMMXY_AUTO_DWLSSVMRegression.__set_kernel(
            self.KMM_kernel, **params)
        if isinstance(
                X_source,
            (pd.DataFrame,
             pd.Series)):  #checks if X is an instance of either types
            X_sourceloc = X_source.to_numpy()
        else:
            X_sourceloc = X_source

        if isinstance(y_source, (pd.DataFrame, pd.Series)):
            y_sourceloc = y_source.to_numpy()
        else:
            y_sourceloc = y_source
        if X_sourceloc.ndim == 2:
            self.x_source = X_sourceloc
        else:
            self.x_source = X_sourceloc.reshape(-1, 1)
        if y_sourceloc.ndim == 1:
            self.y_source = y_sourceloc.reshape(-1, 1)
        else:
            self.y_source = y_sourceloc

        self.x_target = None
        self.y_target = None
        self.x = None  
        self.y = None  
        self.coef = None  
        self.KMM_weight = None  
        self.v_weight = None  
        self.e = None  
        self.coef_ = None  
        self.intercept_ = None  

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                         and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "X_source": self.x_source,
            "y_source": self.y_source,
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "KMM_kernel": self.KMM_kernel,
            "kernel": self.kernel,
            "sigma": self.sigma,
            "eta": self.eta,
            'S': self.S,
            "method": self.method
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Args:
            **parameters: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        params = dict()
        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d
        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma
        if self.KMM_kernel == 'KMM_rbf':
            params['eta'] = self.eta
        self.kernel_ = KMMXY_AUTO_DWLSSVMRegression.__set_kernel(
            self.kernel, **params)
        self.KMM_kernel_ = KMMXY_AUTO_DWLSSVMRegression.__set_kernel(
            self.KMM_kernel, **params)
        return self

    @staticmethod
    def __set_kernel(name, **params):
        """Set the kernel function based on the specified name and parameters."""
        def linear(xi, xj):
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            return ((np.dot(xi, xj.T)) / c + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            from scipy.spatial.distance import cdist
            if xi.ndim == 2 and xi.ndim == xj.ndim:
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) / (2 * (sigma**2)))
            elif xi.ndim < 2 and xj.ndim < 3:
                ax = len(xj.shape) - 1
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax) - 2 * np.dot(xi, xj.T)) / (2 * (sigma**2)))
            else:
                raise ValueError("The rbf kernel is not suited for arrays with rank >2")

        def KMM_rbf(xi, xj, eta=params.get('eta', 1.0)):
            from scipy.spatial.distance import cdist

            if (xi.ndim == 2 and xi.ndim == xj.ndim):  # both are 2D matrices
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean')) /
                                (2 * (eta**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape) - 1  #compensate for python zero-base
                return np.exp(-(np.dot(xi, xi) +
                                (xj**2).sum(axis=ax) - 2 * np.dot(xi, xj.T)) /
                                (2 * (eta**2)))
            else:
                message = "The KMM_rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)

        kernels = {
            'linear': linear,
            'poly': poly,
            'rbf': rbf,
            'KMM_rbf': KMM_rbf
        }

        if kernels.get(name) is not None:
            return kernels[name]
        else:  #unknown kernel: crash and burn?
            message = "Kernel " + name + " is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)
    def fit(self, X_target, y_target):
        """
        Fit the KMMXY_AUTO_DWLSSVMRegression model.

        Args:
            X_target (np.ndarray): Target domain feature matrix
            y_target (np.ndarray): Target domain target values

        Returns:
            self: The fitted model
        """
        if isinstance(
                X_target,
            (pd.DataFrame,
             pd.Series)):  #checks if X is an instance of either types
            X_targetloc = X_target.to_numpy()
        else:
            X_targetloc = X_target

        if isinstance(y_target, (pd.DataFrame, pd.Series)):
            y_targetloc = y_target.to_numpy()
        else:
            y_targetloc = y_target
        if X_targetloc.ndim == 2:
            self.x_target = X_targetloc
        else:
            self.x_target = X_targetloc(-1, 1)
        if y_targetloc.ndim == 1:
            self.y_target = y_targetloc.reshape(-1, 1)
        else:
            self.y_target = y_target
        self.x = np.vstack((self.x_source, self.x_target))
        self.y = np.vstack((self.y_source, self.y_target))
        self.xy = np.hstack((self.x, self.y))
        self.kmm_scipy_tuning(self.method)
        self.coef = self.coef / (np.max(self.coef))
        n_s = self.x_target.shape[0]
        self.KMM_weight = np.vstack(((self.coef), np.ones(
            (n_s, 1)))) 
        self.v_weight = np.identity(np.shape(self.y)[0])
        self.__OptimizeParams()
        s = 0
        previous_coef = None
        while s < self.S:
            self.v_weight = self.weights()
            self.__OptimizeParams()
            if previous_coef is not None:
                if abs((self.coef_ - previous_coef).max()) < 0.001:
                    break
            previous_coef = self.coef_
            s = s + 1

    def fix_fit(self, X_train, y_train, sample_weight):
        """
        Fit the model with fixed KMM weights.

        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training target values
            sample_weight (np.ndarray): Fixed sample weights (e.g., KMM weights)

        Returns:
            self: The fitted model
        """
        if isinstance(
                X_train,
            (pd.DataFrame,
             pd.Series)):  #checks if X is an instance of either types
            X_trainloc = X_train.to_numpy()
        else:
            X_trainloc = X_train

        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_trainloc = y_train.to_numpy()
        else:
            y_trainloc = y_train
        if X_trainloc.ndim == 2:
            self.x_train = X_trainloc
        else:
            self.x_train = X_trainloc(-1, 1)
        if y_trainloc.ndim == 1:
            self.y_train = y_trainloc.reshape(-1, 1)
        else:
            self.y_train = y_train
        self.v_weight = np.identity(np.shape(self.y_train)[0])
        self.fix__OptimizeParams(sample_weight)
        s = 0
        previous_coef = None
        while s < self.S:
            self.v_weight = self.weights()
            self.fix__OptimizeParams(sample_weight)
            if previous_coef is not None:
                if abs((self.coef_ - previous_coef).max()) < 0.001:
                    break
            previous_coef = self.coef_
            s = s + 1

    def __OptimizeParams(self):
        """
        Solve the matrix operation to get the coefficients.
        --> equation 3.5 and 3.6 of the book by Suykens
        ==> that is for classification, for regression slightly different cf Dilmen paper 2017

        self.y: 1D array
        self.X: 2D array (with rows the vectors: X[0,:] first vector)

        Set the class parameters:
            - self.intercept_ : intercept
            - self.coef_      : coefficients
            - self.e          : error


        """

        #Regression
        param_fit = {'gamma': self.sigma}
        Omega = pairwise_kernels(self.x,
                                 self.x,
                                 metric=self.kernel,
                                 filter_params=True,
                                 **param_fit)
        Ones = np.array([[1]] * np.shape(
            self.y)[0]) 
        KMMweight_diag = self.KMM_weight.reshape(-1)**-1  
        A_dag = np.linalg.pinv(
            np.block(
                [[0, Ones.T],
                 [
                     Ones, Omega +
                     self.gamma**-1 * self.v_weight * np.diag(KMMweight_diag)
                 ]]))  
        B = np.concatenate((np.array([0]), self.y), axis=None)

        solution = np.dot(A_dag, B)
        self.intercept_ = solution[0]
        self.coef_ = solution[1:]
        v_diagonal = np.diagonal(self.v_weight)
        v_diagonal = v_diagonal.reshape(-1, 1)**-1
        self.e = self.coef_.reshape(
            -1, 1) / (self.gamma * self.KMM_weight.reshape(-1, 1) *
                      v_diagonal.reshape(-1, 1))

    def fix__OptimizeParams(self, sample_weight):
        """
        Solve the matrix operation to get the coefficients.
        --> equation 3.5 and 3.6 of the book by Suykens
        ==> that is for classification, for regression slightly different cf Dilmen paper 2017

        y_train: 1D array
        X_train: 2D array (with rows the vectors: X[0,:] first vector)

        Set the class parameters:
            - intercept_ : intercept
            - coef_      : coefficients
            - e          : error


        """

        #Regression
        param = {'gamma': self.sigma}
        Omega = pairwise_kernels(self.x_train,
                                 self.x_train,
                                 metric=self.kernel,
                                 filter_params=True,
                                 **param)
        Ones = np.array([[1]] * np.shape(
            self.y_train)[0])  # needs to be a 2D 1-column vector, hence [[ ]]
        KMMweight_diag = sample_weight.reshape(-1)**-1  #将权重改为1d方便后面转换成对角阵
        A_dag = np.linalg.pinv(
            np.block(
                [[0, Ones.T],
                 [
                     Ones, Omega +
                     self.gamma**-1 * self.v_weight * np.diag(KMMweight_diag)
                 ]]))  #need to check if the matrix is OK--> y.T parts
        B = np.concatenate((np.array([0]), self.y_train), axis=None)

        solution = np.dot(A_dag, B)
        self.intercept_ = solution[0]
        self.coef_ = solution[1:]
        v_diagonal = np.diagonal(self.v_weight)
        v_diagonal = v_diagonal.reshape(-1, 1)**-1
        self.e = self.coef_.reshape(
            -1, 1) / (self.gamma * sample_weight.reshape(-1, 1) *
                      v_diagonal.reshape(-1, 1))       

    def fit_SVR(self,
                X_target: np.ndarray,
                y_target: np.ndarray,
                method='hyperopt'):
        """
        Fit an SVR model with various hyperparameter optimization methods.

        Args:
            X_target (np.ndarray): Target domain feature matrix
            y_target (np.ndarray): Target domain target values
            method (str): Hyperparameter optimization method ('gs', 'hyperopt', or 'pso')

        Returns:
            self: The fitted model
        """
        if isinstance(
                X_target,
            (pd.DataFrame,
             pd.Series)):  #checks if X is an instance of either types
            X_targetloc = X_target.to_numpy()
        else:
            X_targetloc = X_target

        if isinstance(y_target, (pd.DataFrame, pd.Series)):
            y_targetloc = y_target.to_numpy()
        else:
            y_targetloc = y_target

        if X_targetloc.ndim == 2:
            self.x_target = X_targetloc
        else:
            self.x_target = X_targetloc(-1, 1)
        if y_targetloc.ndim == 1:
            self.y_target = y_targetloc.reshape(-1, 1)
        else:
            self.y_target = y_target
        self.x = np.vstack((self.x_source, self.x_target))
        self.y = np.vstack((self.y_source, self.y_target))
        self.xy = np.hstack((self.x, self.y))
        gammga_XY = np.median(pairwise_distances(self.xy, metric="euclidean"))
        a = list(range(1, 31, 1))
        b = list(range(4, 11, 1))
        c = []
        d = []
        for i in a:
            i = i / 10 * gammga_XY 
            c.append(i)

        for i in b:
            i = i * gammga_XY  
            d.append(i)
        c += d
        #=======KMMXY_weight_AUTO================
        self.kmm_scipy_tuning(self.method, c)
        self.coef = self.coef / (np.max(self.coef))
        n_s = self.x_target.shape[0]
        self.KMM_weight = np.vstack(((self.coef), np.ones(
            (n_s, 1))))  
        if method == 'gs':
            gsmodel = GridSearchCV(SVR(kernel="rbf"),
                                   param_grid={
                                       "C": np.linspace(500, 5000, 10),
                                       "gamma": np.linspace(5e-4, 5e-3, 10)
                                   })
            gsmodel.fit(self.x, self.y, sample_weight=self.KMM_weight.ravel())
            json_str = json.dumps(gsmodel.best_params_)  #dumps
            with open(r'C:\Users\admin\Desktop\GSsvr.txt', 'w') as f:
                f.write(json_str)
                f.close()
            self.model = gsmodel
        elif method == 'hyperopt':
            parameter_space_svr = {
                # loguniform表示该参数取对数后符合均匀分布
                'C': hp.loguniform("C", np.log(1e+2), np.log(1e+4)),
                'kernel': hp.choice('kernel', ['rbf', 'poly']),
                'gamma': hp.loguniform("gamma", np.log(1e-3), np.log(1e+1)),
                # 'scale': hp.choice('scale', [0, 1]),
                'normalize': hp.choice('normalize', [0, 1])
            }

            def normalize(x):
                sc = StandardScaler()
                X_std = sc.fit_transform(x)
                return X_std, sc

            def scale(x):
                sc = MinMaxScaler()
                X_scale = sc.fit_transform(x)
                return X_scale, sc

            def function(args):
                X_ = self.x[:]
                Y_ = self.y[:]
                if 'normalize' in args:
                    if args['normalize'] == 1:
                        X_, _ = normalize(X_)
                        Y_, _ = normalize(Y_)
                    del args['normalize']
                clf = SVR(**args)
                weighted_scores = cross_val_scores_weighted(
                    clf,
                    X_,
                    Y_,
                    self.KMM_weight,
                    cv=5,
                    metrics=[mean_squared_error])
                score = np.array(weighted_scores).mean()
                return {'loss': score, 'args': args, 'status': STATUS_OK}

            bayes_trials = Trials()
            best = fmin(function,
                        parameter_space_svr,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=bayes_trials)
            kernel_list = ['rbf', 'poly']
            best["kernel"] = kernel_list[best["kernel"]]
            json_str1 = json.dumps(best)
            self.svr_param = best
            if 'normalize' in best:
                if best['normalize'] == 1:
                    X_std, self.sc_x = normalize(self.x)
                    Y_std, self.sc_y = normalize(self.y)
                    del best['normalize']
                    hyperclf = SVR(**best)
                    hyperclf.fit(X_std,
                                 Y_std.ravel(),
                                 sample_weight=self.KMM_weight.ravel())
                    self.model = hyperclf
                else:
                    del best['normalize']
                    hyperclf = SVR(**best)
                    hyperclf.fit(self.x,
                                 self.y.ravel(),
                                 sample_weight=self.KMM_weight.ravel())
                    self.model = hyperclf
            else:
                hyperclf = SVR(**best)
                hyperclf.fit(self.x,
                             self.y.ravel(),
                             sample_weight=self.KMM_weight.ravel())
                self.model = hyperclf
        else:
            raise ValueError('unknown method')
    def get_KMMweight(
        self,
        X_target: np.ndarray,
        y_target: np.ndarray,
    ):
        """
        Get KMM weights for given target data.

        Args:
            X_target (np.ndarray): Target domain feature matrix
            y_target (np.ndarray): Target domain target values

        Returns:
            np.ndarray: Computed KMM weights
        """
        if isinstance(
                X_target,
            (pd.DataFrame,
             pd.Series)):  #checks if X is an instance of either types
            X_targetloc = X_target.to_numpy()
        else:
            X_targetloc = X_target

        if isinstance(y_target, (pd.DataFrame, pd.Series)):
            y_targetloc = y_target.to_numpy()
        else:
            y_targetloc = y_target
        if X_targetloc.ndim == 2:
            self.x_target = X_targetloc
        else:
            self.x_target = X_targetloc(-1, 1)
        if y_targetloc.ndim == 1:
            self.y_target = y_targetloc.reshape(-1, 1)
        else:
            self.y_target = y_target
        self.x = np.vstack((self.x_source, self.x_target))
        self.y = np.vstack((self.y_source, self.y_target))
        self.xy = np.hstack((self.x, self.y))
        gammga_XY = np.median(pairwise_distances(self.xy, metric="euclidean"))
        a = list(range(1, 31, 1))
        b = list(range(4, 11, 1))
        c = []
        d = []
        for i in a:
            i = i / 10 * gammga_XY  
            c.append(i)

        for i in b:
            i = i * gammga_XY  
            d.append(i)
        c += d
        #=======KMMXY_weight_AUTO================
        self.kmm_scipy_tuning(self.method, c)
        self.coef = self.coef / (np.max(self.coef))
        n_s = self.x_target.shape[0]
        KMM_weight = np.vstack(((self.coef), np.ones((n_s, 1))))  #源域+目标域总权重
        return KMM_weight
    def weights(self):
        """Compute the weights for the weighted LSSVM."""
        c1, c2 = 2.5, 3
        e = self.e
        m = np.shape(e)[0]
        v = np.zeros((m, 1))
        v1 = np.eye(m)
        shang = np.zeros((m, 1))
        s = median_abs_deviation(e, scale='normal', axis=None)
        for j in range(m):
            shang[j, 0] = abs(e[j, 0] / s)
        for x in range(m):
            if shang[x, 0] <= c1:
                v[x, 0] = 1.0
            elif c1 < shang[x, 0] <= c2:
                v[x, 0] = (c2 - shang[x, 0]) / (c2 - c1)
            else:
                v[x, 0] = 0.0001
            v1[x, x] = 1 / float(v[x, 0])
        return v1
    def fit_featurewise_kernel_mean_matching(self, eta=1, B=1000, eps=None, lmbd=1):
        """
        Perform feature-wise kernel mean matching.

        Args:
            eta (float): KMM kernel parameter
            B (float): Upper bound on KMM weights
            eps (float): Tolerance for weight sum
            lmbd (float): Regularization parameter

        Returns:
            np.ndarray: KMM weights
        """
        _each_coef = self._get_coef(self.x_target, self.x_source,
                                    self.y_target, self.y_source, B, eps, lmbd,
                                    eta).ravel()
        # _coefs[:,ii] = _each_coef
        return _each_coef.reshape(-1, 1)

    def _get_coef(self, X_t, X_s, Y_t, Y_s, B, eps, lmbd, eta):
        """
        Compute KMM coefficients using quadratic programming.

        Args:
            X_t, Y_t (np.ndarray): Target domain data
            X_s, Y_s (np.ndarray): Source domain data
            B (float): Upper bound on KMM weights
            eps (float): Tolerance for weight sum
            lmbd (float): Regularization parameter
            eta (float): KMM kernel parameter

        Returns:
            np.ndarray: KMM coefficients
        """
        n_t = X_t.shape[0]
        n_s = X_s.shape[0]
        Xy_source = np.hstack((X_s, Y_s))
        Xy_target = np.hstack((X_t, Y_t))
        if eps == None:
            eps = math.sqrt((n_s - 1)) / math.sqrt(n_s)
        if self.KMM_kernel == 'lin':
            K = np.dot(X_s, X_s.T)
            kappa = np.sum(np.dot(X_s, X_t) * float(n_s) / float(n_t), axis=1)
        elif self.KMM_kernel == 'KMM_rbf':
            params = {'gamma': eta}
            K = pairwise_kernels(Xy_source,
                                 Xy_source,
                                 metric='rbf',
                                 filter_params=True,
                                 **params)
            kappa = np.sum(pairwise_kernels(Xy_source,
                                            Xy_target,
                                            metric='rbf',
                                            filter_params=True,
                                            **params),
                           axis=1) * float(n_s) / float(n_t)

        else:
            raise ValueError('unknown kernel')
        K = K + lmbd * np.eye(n_s)
        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, n_s)), -np.ones((1, n_s)),
                         np.eye(n_s), -np.eye(n_s)])
        h = matrix(np.r_[n_s * (1 + eps), n_s * (eps - 1), B * np.ones(
            (n_s, )),
                         np.zeros((n_s, ))])

        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])

        return coef
    
    def predict_SVR(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the regression values for a set of feature vectors

        Parameters:
            - X: ndarray of feature vectors (max: 2D), 1 per row if more than one.

        """
        # Ker = self.kernel_(X, self.x) #second component should be the array of training vectors
        # Y = np.dot(self.coef_, Ker.T) + self.intercept_
        if 'normalize' in self.svr_param:
            if self.svr_param['normalize'] == 1:
                X = self.sc_x.transform(X)
                Y = self.sc_y.inverse_transform(
                    self.model.predict(X.reshape(-1, 1)))
            else:
                Y = self.model.predict(X.reshape(-1, 1))
        else:
            Y = self.model.predict(X.reshape(-1, 1))
        return Y
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the regression values for a set of feature vectors

        Parameters:
            - X: ndarray of feature vectors (max: 2D), 1 per row if more than one.

        """
        param_predict = {'gamma': self.sigma}
        Ker = pairwise_kernels(
            X, self.x, metric=self.kernel, filter_params=True, **param_predict
        )  #second component should be the array of training vectors
        Y = np.dot(self.coef_, Ker.T) + self.intercept_
        return Y

    def fix_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fixed fit model.

        Args:
            - X: ndarray of feature vectors (max: 2D), 1 per row if more than one.

        """
        param_predict = {'gamma': self.sigma}
        Ker = pairwise_kernels(
            X,
            self.x_train,
            metric=self.kernel,
            filter_params=True,
            **param_predict
        )  #second component should be the array of training vectors
        Y = np.dot(self.coef_, Ker.T) + self.intercept_
        return Y
    @staticmethod
    def computeKernelWidth(data):
        """
        Compute the kernel width as the median of pairwise distances.

        Args:
            data (np.ndarray): Input data

        Returns:
            float: Computed kernel width
        """
        dist = []
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                dist.append(np.sqrt(np.sum((np.array(data[[i], ]) - np.array(data[[j], ]))**2)))
        return np.median(np.array(dist))

    def kmm_tuning(self, gammab=None):
        """
        Tune KMM parameters using J-score.

        Args:
            gammab (list): List of gamma values to try

        Returns:
            self: The instance with tuned parameters
        """
        Jmin = float('inf')
        beta = []
        best_gammab = 0

        if gammab is None:
            gammab = [1 / float(self.x_source.shape[0]), 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

        for g in gammab:
            betaSource = self.fit_featurewise_kernel_mean_matching(eta=g)
            betaTarget = self.regerssion_beta(self.x_source, self.y_source, self.x_target, self.y_target, betaSource)
            J = self.computeJ(betaSource, betaTarget)
            if J < Jmin:
                Jmin = J
                beta = list(betaSource)
                best_gammab = g

        self.coef = beta
        self.best_gammab = best_gammab
        return self
    
    @staticmethod
    def computeJ(betaSource, betaTarget):
        """
        Compute J score for parameter tuning of KMM.

        Args:
            betaSource (np.ndarray): Source domain weights
            betaTarget (np.ndarray): Target domain weights

        Returns:
            float: Computed J score
        """
        betaSource = betaSource.reshape(-1).tolist()
        betaTarget = betaTarget.reshape(-1).tolist()
        tr = sum([i**2 for i in betaSource])
        te = sum(betaTarget)
        return ((1 / float(len(betaSource))) * tr) - ((2 / float(len(betaTarget))) * te)

    @staticmethod
    def regerssion_beta(X_source, y_source, X_target, y_target, betaSource):
        """
        Train a regression model to predict target domain weights.

        Args:
            X_source, y_source (np.ndarray): Source domain data
            X_target, y_target (np.ndarray): Target domain data
            betaSource (np.ndarray): Source domain weights

        Returns:
            np.ndarray: Predicted target domain weights
        """
        model = GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 3, 4), "gamma": np.logspace(-2, 2, 5)})
        Xy_source = np.hstack((X_source, y_source))
        Xy_target = np.hstack((X_target, y_target))
        model.fit(Xy_source, betaSource)
        beta = model.predict(Xy_target)
        return beta

    @staticmethod 
    def computeWD(X_source, y_source, X_target, y_target, betaSource):
        """
        Compute Wasserstein distance between source and target domains.

        Args:
            betaSource (np.ndarray): Source domain weights

        Returns:
            float: Computed Wasserstein distance
        """
        Xy_source = torch.from_numpy(np.hstack((X_source, y_source)))
        Xy_target = torch.from_numpy(np.hstack((X_target, y_target)))
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        dist, P, C = sinkhorn(betaSource * Xy_source, Xy_target)
        return dist

    def computeWD(self, betaSource):
        Xy_source = torch.from_numpy(np.hstack((self.x_source, self.y_source)))
        Xy_target = torch.from_numpy(np.hstack((self.x_target, self.y_target)))
        betaSource = torch.from_numpy(betaSource)
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=300, reduction=None)
        dist, P, C = sinkhorn(betaSource * Xy_source, Xy_target)
        return dist.item()
    def computeWDloss(self, betaSource):
        """
        Compute Wasserstein distance loss between source and target domains.

        Args:
            betaSource (np.ndarray): Source domain weights

        Returns:
            float: Computed Wasserstein distance loss
        """
        Xy_source = torch.from_numpy(np.hstack((self.x_source, self.y_source)))
        Xy_target = torch.from_numpy(np.hstack((self.x_target, self.y_target)))
        betaSource = torch.from_numpy(betaSource)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        L = loss(betaSource * Xy_source, Xy_target)
        return L.item()

    def auto_tuningloss(self, beta):
        """
        Compute the loss for auto-tuning KMM parameters.

        Args:
            beta (float): KMM parameter

        Returns:
            float: Computed loss
        """
        betaSource = self.fit_featurewise_kernel_mean_matching(eta=beta)
        return self.computeWDloss(betaSource.reshape(-1, 1))

    def kmm_scipy_tuning(self, method='minimize_scalar', gammab: tuple = None):
        """
        Tune KMM parameters using various optimization methods.

        Args:
            method (str): Optimization method to use
            gammab (tuple): Bounds for the gamma parameter (used in 'brute' method)

        Returns:
            self: The instance with tuned parameters
        """
        x0 = np.asarray((0.1))  
        if method == 'minimize_scalar':
            res = optimize.minimize_scalar(self.auto_tuningloss,
                                           bounds=(0, 100),
                                           method='bounded')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                eta=self.best_gammab)
        elif method == 'SLSQP':
            res = optimize.minimize(self.auto_tuningloss,
                                    x0,
                                    bounds=[(0, None)],
                                    method='SLSQP')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                eta=self.best_gammab)
        elif method == 'basinhopping':
            res = optimize.basinhopping(self.auto_tuningloss, x0)
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                eta=self.best_gammab)
        elif method == 'brute':
            if gammab is None:
                gammab = (0, 10, 0.5)
            self.best_gammab = optimize.brute(self.auto_tuningloss, (gammab, ),
                                              finish=None)
            self.coef = self.fit_featurewise_kernel_mean_matching(
                eta=self.best_gammab)
        elif method == 'kernelwidth':
            self.best_gammab = np.median(
                pairwise_distances(self.xy, metric="euclidean"))
            self.coef = self.fit_featurewise_kernel_mean_matching(
                eta=self.best_gammab)
        elif method == 'computeJ':
            gammga_XY = np.median(
                pairwise_distances(self.xy, metric="euclidean"))
            a = list(range(1, 31, 1))
            b = list(range(4, 11, 1))
            c = []
            d = []
            for i in a:
                i = i / 10 * gammga_XY  
                c.append(i)

            for i in b:
                i = i * gammga_XY  
                d.append(i)
            c += d
            self.kmm_tuning(c)
        else:
            raise ValueError('unknown method')

        return self
    

