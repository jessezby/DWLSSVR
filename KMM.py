"""
Kernel Mean Matching (KMM) Implementation

This module provides an implementation of Kernel Mean Matching for covariate shift correction.
It includes various methods for parameter tuning and optimization.

"""

import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import check_pairwise_arrays, pairwise_kernels, pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import torch
from geomloss import SamplesLoss
from scipy import optimize
from layers import SinkhornDistance

class kmm:
    def __init__(self, X_target, X_source, Y_target, Y_source, gamma=None, degree=3, coef0=1, kernel_params=None):
        """
        Initialize the KMM object.

        Args:
            X_target (np.ndarray): Target domain feature matrix.
            X_source (np.ndarray): Source domain feature matrix.
            Y_target (np.ndarray): Target domain labels.
            Y_source (np.ndarray): Source domain labels.
            gamma (float, optional): Kernel coefficient for RBF kernel.
            degree (int, optional): Degree of polynomial kernel function.
            coef0 (float, optional): Independent term in kernel function.
            kernel_params (dict, optional): Additional kernel parameters.
        """
        self.X_target, self.X_source = check_pairwise_arrays(X_target, X_source, dtype=float)
        self.Y_target, self.Y_source = check_pairwise_arrays(Y_target, Y_source, dtype=float)
        
        self.x = np.vstack((self.X_source, self.X_target))
        self.y = np.vstack((self.Y_source, self.Y_target))
        self.xy = np.hstack((self.x, self.y))

        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.coef = None
        self.best_gammab = None

    def fit_featurewise_kernel_mean_matching(self, eta, kern='rbf', B=1000, eps=None, lmbd=1):
        """
        Perform featurewise kernel mean matching.

        Args:
            eta (float): Kernel parameter.
            kern (str): Kernel type ('rbf' or 'lin').
            B (float): Upper bound on the importance weights.
            eps (float): Tolerance parameter.
            lmbd (float): Regularization parameter.

        Returns:
            np.ndarray: Computed coefficients.
        """
        self.kernel = kern
        coef = self._get_coef(self.X_target, self.X_source, self.Y_target, self.Y_source, eta, kern, B, eps, lmbd).ravel()
        return coef.reshape(-1, 1)

    def _get_coef(self, X_t, X_s, Y_t, Y_s, eta, kern, B, eps, lmbd):
        """Helper method to compute KMM coefficients."""
        n_s = X_s.shape[0]
        n_t = X_t.shape[0]
        Xy_source = np.hstack((X_s, Y_s))
        Xy_target = np.hstack((X_t, Y_t))
        
        eps = eps or (n_s - 1) / np.sqrt(n_s)
        
        if kern == 'lin':
            K = np.dot(X_s, X_s.T)
            kappa = np.sum(np.dot(X_s, X_t.T) * float(n_s) / float(n_t), axis=1)
        elif kern == 'rbf':
            params = {'gamma': eta}
            K = pairwise_kernels(Xy_source, Xy_source, metric='rbf', filter_params=True, **params)
            kappa = np.sum(pairwise_kernels(Xy_source, Xy_target, metric='rbf', filter_params=True, **params), axis=1) * float(n_s) / float(n_t)
        else:
            raise ValueError('Unknown kernel')
        
        K = K + lmbd * np.eye(n_s)
        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, n_s)), -np.ones((1, n_s)), np.eye(n_s), -np.eye(n_s)])
        h = matrix(np.r_[n_s * (1 + eps), n_s * (eps - 1), B * np.ones((n_s,)), np.zeros((n_s,))])
        
        sol = solvers.qp(K, -kappa, G, h)
        return np.array(sol['x'])

    @staticmethod
    def regerssion_beta(X_source, y_source, X_target, y_target, betaSource):
        """Perform regression to predict beta for target domain."""
        model = GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 3, 4), "gamma": np.logspace(-2, 2, 5)})
        Xy_source = np.hstack((X_source, y_source))
        Xy_target = np.hstack((X_target, y_target))
        model.fit(Xy_source, betaSource.ravel())
        return model.predict(Xy_target)

    @staticmethod
    def computeJ(betaSource, betaTarget):
        """Compute J score for KMM parameter tuning."""
        betaSource = betaSource.reshape(-1)
        betaTarget = betaTarget.reshape(-1)
        tr = np.sum(betaSource**2)
        te = np.sum(betaTarget)
        return (tr / len(betaSource)) - (2 * te / len(betaTarget))

    def tuning(self, gammab=None):
        """Perform KMM parameter tuning."""
        gammab = gammab or [1/self.X_source.shape[0], 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        
        Jmin = float('inf')
        best_beta = None
        best_g = None
        
        for g in gammab:
            betaSource = self.fit_featurewise_kernel_mean_matching(eta=g, kern='rbf')
            betaTarget = self.regerssion_beta(self.X_source, self.Y_source, self.X_target, self.Y_target, betaSource)
            J = self.computeJ(betaSource, betaTarget)
            
            if J < Jmin:
                Jmin = J
                best_beta = betaSource
                best_g = g
        
        self.Jmin = Jmin
        self.coef = best_beta
        self.best_gammab = best_g
        return self

    def auto_tuningloss(self, beta):
        """Compute loss for automatic tuning."""
        betaSource = self.fit_featurewise_kernel_mean_matching(kern='rbf', eta=beta)
        return abs(self.computeWDloss(betaSource))

    def kmm_scipy_tuning(self, method='minimize_scalar', gammab=None):
        """Perform KMM tuning using scipy optimization methods."""
        if method == 'minimize_scalar':
            res = optimize.minimize_scalar(self.auto_tuningloss, bounds=(0, 100), method='bounded')
        elif method == 'SLSQP':
            res = optimize.minimize(self.auto_tuningloss, x0=0.1, method='SLSQP')
        elif method == 'basinhopping':
            res = optimize.basinhopping(self.auto_tuningloss, x0=0.1)
        elif method == 'brute':
            gammab = gammab or (0, 10, 0.5)
            res = optimize.brute(self.auto_tuningloss, (gammab,), finish=None)
        else:
            raise ValueError('Unknown method')
        
        self.best_gammab = res.x if hasattr(res, 'x') else res
        self.coef = self.fit_featurewise_kernel_mean_matching(kern='rbf', eta=self.best_gammab)
        return self

    def computeWDloss(self, betaSource):
        """Compute Wasserstein distance loss."""
        Xy_source = torch.from_numpy(np.hstack((self.X_source, self.Y_source)))
        Xy_target = torch.from_numpy(np.hstack((self.X_target, self.Y_target)))
        betaSource = torch.from_numpy(betaSource)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        return loss(betaSource * Xy_source, Xy_target).item()

# Additional methods like wigth_plotdata, kmm_kernelwidth_tunning, etc. can be implemented similarly