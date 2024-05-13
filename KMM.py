

from itertools import tee
from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
import numpy as np
import math
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import linear_model
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import torch
from geomloss import SamplesLoss
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy import optimize
from layers import SinkhornDistance
from scipy.optimize import minimize_scalar
from sklearn.metrics.pairwise import pairwise_distances

class kmm:
    def __init__(self,
                 X_target,
                 X_source,
                 Y_target,
                 Y_source,
                 gamma=None,
                 degree=3,
                 coef0=1,
                 kernel_params=None):
        X_target = np.atleast_1d(X_target)
        X_source = np.atleast_1d(X_source)
        Y_target = np.atleast_1d(Y_target)
        Y_source = np.atleast_1d(Y_source)
        if (X_target.shape[0] == X_target.size):
            X_target = X_target.reshape(-1, 1)
        if (Y_target.shape[0] == Y_target.size):
            Y_target = Y_target.reshape(-1, 1)
        if Y_source.shape[0] == Y_source.size:
            Y_source = Y_source.reshape(-1, 1)
        X_target, X_source = check_pairwise_arrays(X_target,
                                                   X_source,
                                                   dtype=float)
        Y_target, Y_source = check_pairwise_arrays(Y_target,
                                                   Y_source,
                                                   dtype=float)

        self.X_target = X_target
        self.X_source = X_source
        self.Y_target = Y_target
        self.Y_source = Y_source
        self.x = np.vstack((self.X_source, self.X_target))
        self.y = np.vstack((self.Y_source, self.Y_target))
        self.xy = np.hstack((self.x, self.y))

        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        return None

    def fit_kernel_mean_matching(self,
                                 kern='lin',
                                 sigma=1,
                                 B=1.0,
                                 eps=None,
                                 lmbd=1):
        _X_target = self.X_target
        _X_source = self.X_source
        _coef = self._get_coef(_X_target, _X_source, kern, sigma, B, eps, lmbd)
        self.coef = _coef
        return self

    def _get_coef(self, X_t, X_s, Y_t, Y_s, eta, kern, B, eps, lmbd):

        n_t = X_t.shape[0]
        n_s = X_s.shape[0]
        Xy_source = np.hstack((X_s, Y_s))
        Xy_target = np.hstack((X_t, Y_t))
        if eps == None:
            eps = math.sqrt((n_s - 1)) / math.sqrt(n_s)
        if kern == 'lin':
            K = np.dot(X_s, X_s.T)
            kappa = np.sum(np.dot(X_s, X_t) * float(n_s) / float(n_t), axis=1)
        elif kern == 'rbf':
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

    def _get_kernel(self, X, Y=None, indicted_gamma=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                'gamma': self.gamma,
                'degree': self.degree,
                'coef0': self.coef0
            }
        if indicted_gamma:  # if not indicated, use self.gamma
            params['gamma'] = indicted_gamma
        return pairwise_kernels(X,
                                Y,
                                metric=self.kernel,
                                filter_params=True,
                                **params)

    def fit_featurewise_kernel_mean_matching(self,
                                             eta,
                                             kern='rbf',
                                             B=1000,
                                             eps=None,
                                             lmbd=1):
        self.kernel = kern
        _each_coef = self._get_coef(self.X_target, self.X_source,
                                    self.Y_target, self.Y_source, eta, kern, B,
                                    eps, lmbd).ravel()

        return _each_coef.reshape(-1, 1)
    @staticmethod
    def regerssion_beta(X_source, y_source, X_target, y_target, betaSource):
        model = GridSearchCV(SVR(),
                             param_grid={
                                 "C": np.logspace(0, 3, 4),
                                 "gamma": np.logspace(-2, 2, 5)
                             })
        Xy_source = np.hstack((X_source, y_source))
        Xy_target = np.hstack((X_target, y_target))
        model.fit(Xy_source, betaSource.ravel())
        beta = model.predict(Xy_target)
        return beta
    @staticmethod
    def computeJ(betaSource, betaTarget):
        betaSource = betaSource.reshape(-1).tolist()
        betaTarget = betaTarget.reshape(-1).tolist()
        tr = sum([i**2 for i in betaSource])
        te = sum(betaTarget)
        return ((1 / float(len(betaSource))) * tr) - (
            (2 / float(len(betaTarget))) * te)
    def tuning(self, gammab: list = None):
        Jmin = 0
        beta = []
        best_g = 0
        if gammab is None:
            gammab = [
                1 / float(self.X_source.shape[0]), 0.0001, 0.0005, 0.001,
                0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10
            ] 
        for g in gammab:
            betaSource = self.fit_featurewise_kernel_mean_matching(kern='rbf',
                                                                   eta=g)
            betaTarget = kmm.regerssion_beta(self.X_source, self.Y_source,
                                             self.X_target, self.Y_target,
                                             betaSource)
            J = kmm.computeJ(betaSource, betaTarget)
            if len(beta) == 0:
                Jmin = J
                beta = list(betaSource)
            elif Jmin > J:
                Jmin = J
                beta = list(betaSource)
                best_g = g
        self.Jmin = Jmin
        self.coef = beta
        self.best_gammab = best_g
        return self

    def wigth_plotdata(self, method='J', gammab: list = None):
        wigth_all = []
        if gammab is None:
            gammab = np.arange(0.1, 20, 0.1)
        if method == 'J':
            for g in gammab:
                betaSource = self.fit_featurewise_kernel_mean_matching(
                    kern='rbf', eta=g)
                betaTarget = kmm.regerssion_beta(self.X_source, self.Y_source,
                                                 self.X_target, self.Y_target,
                                                 betaSource)
                J = kmm.computeJ(betaSource, betaTarget)
                wigth_all.append(J)
            wigth_data = list(zip(gammab, wigth_all))
        elif method == 'WD':
            for g in gammab:
                WD = self.auto_tuningloss(g)
                wigth_all.append(WD)
            wigth_data = list(zip(gammab, wigth_all))
        else:
            raise ValueError('unknown method')

        return np.array(wigth_data)

    def auto_computeJ_loss(self, beta):
        betaSource = self.fit_featurewise_kernel_mean_matching(kern='rbf',
                                                               eta=beta)

        betaTarget = kmm.regerssion_beta(self.X_source, self.Y_source,
                                         self.X_target, self.Y_target,
                                         betaSource)
        J = kmm.computeJ(betaSource, betaTarget)
        return J

    def scipy_computJ_tuning(self, method='minimize_scalar'):
        x0 = np.asarray((0.1)) 
        if method == 'minimize_scalar':
            res = minimize_scalar(self.auto_computeJ_loss,
                                  bounds=(0, 100),
                                  method='bounded')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'SLSQP':
            res = optimize.minimize(self.auto_computeJ_loss,
                                    x0,
                                    method='SLSQP')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'basinhopping':
            res = optimize.basinhopping(self.auto_computeJ_loss, x0)
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'brute':
            if gammab is None:
                gammab = (0, 10, 0.5)
            self.best_gammab = optimize.brute(self.auto_computeJ_loss,
                                              (gammab, ),
                                              finish=None)
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        else:
            raise ValueError('unknown method')
        return self
    
    @staticmethod
    def computeKernelWidth(data):
        dist = []
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                dist.append(
                    np.sqrt(
                        np.sum((np.array(data[[i], ]) -
                                np.array(data[[j], ]))**2)))

        return np.median(np.array(dist))

    def computeWD(self, betaSource):
        Xy_source = torch.from_numpy(np.hstack((self.X_source, self.Y_source)))
        Xy_target = torch.from_numpy(np.hstack((self.X_target, self.Y_target)))
        betaSource = torch.from_numpy(betaSource)
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=300, reduction=None)
        dist, P, C = sinkhorn(betaSource * Xy_source, Xy_target)
        return dist.item()

    def computeWDloss(self, betaSource):
        Xy_source = torch.from_numpy(np.hstack((self.X_source, self.Y_source)))
        Xy_target = torch.from_numpy(np.hstack((self.X_target, self.Y_target)))
        betaSource = torch.from_numpy(betaSource)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        L = loss(betaSource * Xy_source, Xy_target)
        return abs(L.item())

    def kmm_computeJ_gs_tuning(self):
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
        self.tuning(c)

    def kmm_kernelwidth_tunning(self):
        self.best_gammab = np.median(
            pairwise_distances(self.xy, metric="euclidean"))
        self.coef = self.fit_featurewise_kernel_mean_matching(
            kern='rbf', eta=self.best_gammab)

    def kmm_tuning(self, gammab: list = None):
        Jmin = 0
        beta = []
        best_gammab = 0
        if gammab is None:
            gammab = [
                1 / float(self.X_source.shape[0]), 0.0001, 0.0005, 0.001,
                0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10
            ]  
        for g in gammab:
            betaSource = self.fit_featurewise_kernel_mean_matching(kern='rbf',
                                                                   eta=g)
            betaTarget = kmm.regerssion_beta(self.X_source, self.Y_source,
                                             self.X_target, self.Y_target,
                                             betaSource)
            J = kmm.computeJ(betaSource, betaTarget)
            if len(beta) == 0:
                Jmin = J
                beta = list(betaSource)
            elif Jmin > J:
                Jmin = J
                beta = list(betaSource)
                best_gammab = g
        self.WDmin = Jmin
        self.coef = beta
        self.best_gammab = best_gammab
        return self

    def auto_tuningloss(self, beta):
        betaSource = self.fit_featurewise_kernel_mean_matching(kern='rbf',
                                                               eta=beta)
        WD1 = self.computeWDloss(betaSource)
        return WD1

    def kmm_auto_tuning(self, gammab: tuple = None):
        if gammab is None:
            gammab = (0, 10, 1)
        self.best_gammab = optimize.brute(self.auto_tuningloss, (gammab, ),
                                          finish=None)
        self.coef = self.fit_featurewise_kernel_mean_matching(
            kern='rbf', eta=self.best_gammab)
        return self

    def kmm_scipy_tuning(self, method='minimize_scalar', gammab: tuple = None):
        x0 = np.asarray((0.1))  
        if method == 'minimize_scalar':
            res = minimize_scalar(self.auto_tuningloss,
                                  bounds=(0, 100),
                                  method='bounded')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'SLSQP':
            res = optimize.minimize(self.auto_tuningloss, x0, method='SLSQP')
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'basinhopping':
            res = optimize.basinhopping(self.auto_tuningloss, x0)
            self.best_gammab = res.x
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        elif method == 'brute':
            if gammab is None:
                gammab = (0, 10, 0.5)
            self.best_gammab = optimize.brute(self.auto_tuningloss, (gammab, ),
                                              finish=None)
            self.coef = self.fit_featurewise_kernel_mean_matching(
                kern='rbf', eta=self.best_gammab)
        else:
            raise ValueError('unknown method')
        return self