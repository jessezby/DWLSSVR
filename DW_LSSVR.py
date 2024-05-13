from pickle import NONE
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np 
import math
from cvxopt import matrix, solvers


class DW_LSSVMRegression(BaseEstimator, RegressorMixin):
    def __init__(self, gamma: float = 1.0, kernel: str = None, KMM_kernel:str = None,c: float = 1.0,
                 d: float = 2, sigma: float = 1.0,eta:float=1.0,S:int =10):
        self.gamma = gamma 
        self.c = c
        self.d = d
        self.sigma = sigma 
        self.eta=eta       
        self.S=S           
        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel 
        if KMM_kernel is None:
            self.KMM_kernel = 'KMM_rbf'
        else:
            self.KMM_kernel = KMM_kernel  

        params = dict()
        if kernel == 'poly':
            params['c'] = c
            params['d'] = d
        elif kernel == 'rbf':
            params['sigma'] = sigma
        if KMM_kernel=='KMM_rbf':    
            params['eta']=eta

        self.kernel_ = DW_LSSVMRegression.__set_kernel(self.kernel, **params)
        self.KMM_kernel_=DW_LSSVMRegression.__set_kernel(self.KMM_kernel, **params)

 
        self.x_source = None
        self.y_source = None
        self.x_target=None
        self.y_target=None
        self.x=None        
        self.y=None         
        self.coef=None        
        self.KMM_weight=None 
        self.v_weight=None   
        self.e=None         
        self.coef_ = None       
        self.intercept_ = None   

    def get_params(self, deep=True):
        return {"c": self.c, "d": self.d, "gamma": self.gamma,"KMM_kernel":self.KMM_kernel,
                "kernel": self.kernel, "sigma":self.sigma,"eta":self.eta,'S':self.S}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)
        params = dict()
        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d
        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma
        if self.KMM_kernel=='KMM_rbf':    
            params['eta']=self.eta
        self.kernel_ = DW_LSSVMRegression.__set_kernel(self.kernel, **params)
        self.KMM_kernel_=DW_LSSVMRegression.__set_kernel(self.KMM_kernel, **params)
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
            return ((np.dot(xi, xj.T))/c  + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            from scipy.spatial.distance import cdist

            if (xi.ndim == 2 and xi.ndim == xj.ndim): # both are 2D matrices
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean'))/(2*(sigma**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape)-1 #compensate for python zero-base
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax)
                                - 2*np.dot(xi, xj.T))/(2*(sigma**2)))
            else:
                message = "The rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)
        def KMM_rbf(xi, xj, eta=params.get('eta', 1.0)):
            from scipy.spatial.distance import cdist


            if (xi.ndim == 2 and xi.ndim == xj.ndim): 
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean'))/(2*(eta**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape)-1 
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax)
                                - 2*np.dot(xi, xj.T))/(2*(eta**2)))
            else:
                message = "The KMM_rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf,'KMM_rbf':KMM_rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else: 
            message = "Kernel "+name+" is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)

    def __OptimizeParams(self):
        Omega = self.kernel_(self.x, self.x)
        Ones = np.array([[1]]*np.shape(self.y)[0]) 
        KMMweight_diag=1/self.KMM_weight.reshape(-1)
        A_dag = np.linalg.pinv(np.block([
            [0,                           Ones.T                      ],
            [Ones,   Omega + self.gamma**-1 * self.v_weight*np.diag(KMMweight_diag)]
        ]))
        B = np.concatenate((np.array([0]), self.y), axis=None)

        solution = np.dot(A_dag, B)
        self.intercept_ = solution[0]
        self.coef_      = solution[1:]
        v_diagonal=np.diagonal(self.v_weight)
        v_diagonal=1/v_diagonal.reshape(-1,1)
        self.e=self.coef_.reshape(-1,1)/(self.gamma*self.KMM_weight*v_diagonal)


    def fit(self, X_source: np.ndarray, y_source: np.ndarray, X_target: np.ndarray,y_target:np.ndarray):

        if isinstance(X_source, (pd.DataFrame, pd.Series)):
            X_sourceloc = X_source.to_numpy()
        else:
            X_sourceloc = X_source

        if isinstance(y_source, (pd.DataFrame, pd.Series)):
            y_sourceloc = y_source.to_numpy()
        else:
            y_sourceloc = y_source
        if isinstance(X_target, (pd.DataFrame, pd.Series)): 
            X_targetloc = X_target.to_numpy()
        else:
            X_targetloc = X_target

        if isinstance(y_target, (pd.DataFrame, pd.Series)):
            y_targetloc = y_target.to_numpy()
        else:
            y_targetloc = y_target
        if X_sourceloc.ndim == 2:
            self.x_source = X_sourceloc
        else:
             self.x_source = X_sourceloc.reshape(-1,1)
        if y_sourceloc.ndim == 1:
            self.y_source = y_sourceloc.reshape(-1,1)
        else:
            self.y_source=y_sourceloc
        if X_targetloc.ndim == 2:
            self.x_target = X_targetloc
        else:
              self.x_target = X_targetloc(-1,1)
        if y_targetloc.ndim==1:
            self.y_target = y_targetloc.reshape(-1,1)
        else:
            self.y_target=y_target
        self.x=np.vstack((self.x_source,self.x_target))
        self.y=np.vstack((self.y_source,self.y_target))
        self.coef=self.fit_featurewise_kernel_mean_matching(eta=0.5)
        self.coef=self.coef/(np.max(self.coef))
        self.KMM_weight=np.vstack(((self.coef),np.ones(self.x_target.shape))) 
        self.v_weight=np.identity(np.shape(self.y)[0])
        self.__OptimizeParams()
        s=0
        while s<self.S:
            self.v_weight=self.weights()
            self.__OptimizeParams()
            s=s+1 
           
    def weights(self):
        c1=2.5
        c2=3
        e=self.e
        m=np.shape(e)[0]
        v=np.zeros((m,1))
        v1=np.eye(m)
        q1 = int(m/4.0)
        q3 = int((m*3.0)/4.0)
        e1 = []
        shang = np.zeros((m,1))
        for i in range(m):
            e1.append(e[i,0])
        e1.sort()
        IQR = e1[q3] - e1[q1]
        s = IQR/(2 * 0.6745)
        for j in range(m):
            shang[j,0]=abs(e[j,0]/s)
        for x in range(m):
            if shang[x,0] <= c1:
                v[x,0]=1.0
            if shang[x,0]>c1 and shang[x,0] <=c2:
                v[x,0]=(c2-shang[x,0])/(c2-c1)
            if shang[x,0]>c2:
                v[x,0]=0.0001
        v1[x,x]=1/float(v[x,0])
        return v1

    def fit_featurewise_kernel_mean_matching(self,eta=1, B = 1000, eps = None, lmbd = 1,):
        _coefs = np.zeros(self.x_source.shape)
        for ii in range(self.x_source.shape[1]):
            _Xs = self.x_source[:,ii].reshape(-1,1)
            _Xt = self.x_target[:,ii].reshape(-1,1)
            _Ys = self.y_source[:,ii].reshape(-1,1)
            _Yt = self.y_target[:,ii].reshape(-1,1)
            _each_coef = self._get_coef(_Xt,_Xs,_Yt,_Ys,B, eps, lmbd,eta).ravel()
            _coefs[:,ii] = _each_coef            
        return _coefs
    
    def _get_coef(self, X_t,X_s,Y_t,Y_s, B, eps, lmbd,eta):

        n_t = X_t.shape[0]
        n_s = X_s.shape[0]
        Xy_source = np.hstack((X_s,Y_s))
        Xy_target = np.hstack((X_t,Y_t))
        if eps == None:
            eps = math.sqrt((n_s-1))/math.sqrt(n_s)
        if self.KMM_kernel == 'lin':
            K = np.dot(X_s, X_s.T)
            kappa = np.sum(np.dot(X_s, X_t)*float(n_s)/float(n_t),axis=1)
        elif self.KMM_kernel == 'KMM_rbf':
            K = self.KMM_kernel_(Xy_source,Xy_source,eta) 
            kappa = np.sum(self.KMM_kernel_(Xy_source,Xy_target),axis=1)*float(n_s)/float(n_t)
        else:
            raise ValueError('unknown kernel')
        K = K+lmbd*np.eye(n_s)
        K = matrix(K)
        kappa = matrix(kappa)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        G = matrix(np.r_[np.ones((1,n_s)), -np.ones((1,n_s)), np.eye(n_s), -np.eye(n_s)])
        h = matrix(np.r_[n_s*(1+eps), n_s*(eps-1), B*np.ones((n_s,)), np.zeros((n_s,))])
        
        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])
        
        return coef      
    def predict(self, X: np.ndarray)->np.ndarray:
        Ker = self.kernel_(X, self.x) 
        Y = np.dot(self.coef_, Ker.T) + self.intercept_
        return Y
