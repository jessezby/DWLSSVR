"""
Simulation Example for DWLSSVR

This script demonstrates a simulation process for domain adaptation
using  Double-weight least squares support vector regression(DWLSSVR).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics
import KMM
from DW_LSSVR import KMMXY_AUTO_DWLSSVMRegression
from importance_weighted_cross_validation import lssvrcross_val_scores_weighted

class SimulatePrediction:
    """
    A class to simulate prediction scenarios with domain adaptation.

    Attributes:
        A, B, C (float): Coefficients for source domain function.
        a, b, c (float): Coefficients for target domain function.
    """

    def __init__(self, A=-6, B=0, C=1, a=1, b=1, c=1):
        self.A, self.B, self.C = A, B, C
        self.a, self.b, self.c = a, b, c
        self.x_source = self.y_source = self.x_target = self.y_target = self.x_test = self.y_test = None
        self.std1 = self.std2 = self.std3 = self.std4 = None
        self.x_sourcetarget = self.y_sourcetarget = None
        self.best_gammab = self.Weights_all = None

    def data_generation(self, source_num=500, target_num=10, test_num=200,
                        source_loc=5, source_scale=3, source_noise_scale=200, 
                        target_uniform=5, target_noise_scale=12):
        """
        Generate synthetic data for source, target, and test sets.

        Args:
            source_num (int): Number of source domain samples.
            target_num (int): Number of target domain samples.
            test_num (int): Number of test samples.
            source_loc (float): Mean of source domain distribution.
            source_scale (float): Standard deviation of source domain distribution.
            source_noise_scale (float): Noise scale for source domain.
            target_uniform (float): Range for uniform distribution in target domain.
            target_noise_scale (float): Noise scale for target domain.
        """
        rng1, rng2, rng3 = np.random.RandomState(1), np.random.RandomState(4), np.random.RandomState(3)

        self.x_source = rng1.normal(loc=source_loc, scale=source_scale, size=(source_num, 1))
        self.y_source = self.A * self.x_source + self.B * np.power(self.x_source, 2) + self.C * np.power(self.x_source, 3)
        self.y_source += rng1.normal(0, source_noise_scale, size=self.y_source.shape)

        self.x_target = rng2.uniform(-target_uniform, target_uniform, size=(target_num, 1))
        self.y_target = self.a * self.x_target + self.b * np.power(self.x_target, 2) + self.c * np.power(self.x_target, 3)
        self.y_target += rng2.normal(0, target_noise_scale, size=self.y_target.shape)

        self.x_test = rng3.uniform(-target_uniform, target_uniform, size=(test_num, 1))
        self.y_test = self.a * self.x_test + self.b * np.power(self.x_test, 2) + self.c * np.power(self.x_test, 3)
        self.y_test += rng3.normal(0, target_noise_scale, size=self.y_test.shape)

    def data_normalization(self):
        """Normalize the data using StandardScaler."""
        self.std1, self.std2, self.std3, self.std4 = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()

        self.x_source = self.std1.fit_transform(self.x_source)
        self.y_source = self.std2.fit_transform(self.y_source)
        self.x_target = self.std3.fit_transform(self.x_target)
        self.y_target = self.std4.fit_transform(self.y_target)
        self.x_sourcetarget = np.vstack((self.x_source, self.x_target))
        self.y_sourcetarget = np.vstack((self.y_source, self.y_target))
        self.x_test = self.std3.transform(self.x_test)

    def kmm_weight_wd(self):
        """Compute KMM weights using Wasserstein distance."""
        kmm = KMM.kmm(self.x_target, self.x_source, self.y_target, self.y_source)
        kmm.kmm_scipy_tuning(method='minimize_scalar')
        coef = kmm.coef / np.max(kmm.coef)
        self.best_gammab = kmm.best_gammab
        self.Weights_all = np.vstack((coef, np.ones((self.x_target.shape[0], 1))))

    def hyperopt_tuning(self):
        """Perform hyperparameter tuning using Hyperopt."""
        parameter_space_lssvr = {
            'gamma': hp.loguniform("gamma", np.log(1e+0), np.log(1e+6)),
            'sigma': hp.loguniform("sigma", np.log(1e-3), np.log(1e+3))
        }

        def objective_lssvr(args):
            clf = KMMXY_AUTO_DWLSSVMRegression(self.x_source, self.y_source, kernel="rbf", KMM_kernel="KMM_rbf", **args)
            scores = lssvrcross_val_scores_weighted(clf, self.x_sourcetarget, self.y_sourcetarget, self.Weights_all,
                                                    cv=5, metrics=[mean_squared_error])
            return {'loss': np.array(scores).mean(), 'args': args, 'status': STATUS_OK}

        trials_lssvr = Trials()
        best_lssvr = fmin(objective_lssvr, parameter_space_lssvr, algo=tpe.suggest, max_evals=100, trials=trials_lssvr)
        hyperLSSVR = KMMXY_AUTO_DWLSSVMRegression(self.x_source, self.y_source, kernel="rbf", KMM_kernel="KMM_rbf", **best_lssvr)
        hyperLSSVR.fit(self.x_target, self.y_target)

        Y_HYPERLSSVR = self.std4.inverse_transform(hyperLSSVR.predict(self.x_test).reshape(-1, 1))

        print("Hyperopt LSSVR performance on the test set:")
        print("RMSE:", np.sqrt(metrics.mean_squared_error(self.y_test, Y_HYPERLSSVR)))
        print("R2 score:", r2_score(self.y_test, Y_HYPERLSSVR))

if __name__ == '__main__':
    simulate_process = SimulatePrediction(A=-5, C=2)
    simulate_process.data_generation(source_num=500, test_num=200, source_loc=7, source_scale=3,
                                     target_uniform=4, source_noise_scale=100, target_noise_scale=6, target_num=10)
    simulate_process.data_normalization()
    simulate_process.kmm_weight_wd()
    simulate_process.hyperopt_tuning()