"""
Importance Weighted Cross-Validation

This module provides functions for performing cross-validation with importance weights
for various machine learning models, including support for Leave-One-Out and K-Fold
cross-validation strategies.
"""

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score

def cross_val_scores_weighted(model, X, y, weights, cv=5, metrics=[mean_squared_error]):
    """
    Perform weighted cross-validation.

    Args:
        model: The model to evaluate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        weights (np.ndarray): Sample weights.
        cv (int): Number of folds for cross-validation.
        metrics (list): List of metric functions to evaluate.

    Returns:
        list: List of scores for each metric.
    """
    kf = KFold(n_splits=cv)
    scores = [[] for _ in metrics]
    
    for train_index, test_index in kf.split(X):
        model_clone = clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        
        model_clone.fit(X_train, y_train.ravel(), sample_weight=weights_train.ravel())
        y_pred = model_clone.predict(X_test)
        
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred, sample_weight=weights_test.ravel())
            scores[i].append(score)
    
    return scores

def cross_val_scores_controlweighted(model, X, y, weights, cv=5):
    """
    Perform control-weighted cross-validation.

    Args:
        model: The model to evaluate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        weights (np.ndarray): Sample weights.
        cv (int): Number of folds for cross-validation.

    Returns:
        list: List of scores.
    """
    kf = KFold(n_splits=cv)
    scores = [[]]
    
    for train_index, test_index in kf.split(X):
        model_clone = clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        
        model_clone.fit(X_train, y_train.ravel(), sample_weight=weights_train.ravel())
        y_pred = model_clone.predict(X_test)
        
        weighted_loss = (y_pred.ravel() - y_test.ravel())**2 * weights_test.ravel()
        
        if weights_test.ravel().mean() == 1:
            beta = 0
        else:
            a_i = weighted_loss - weighted_loss.mean()
            b_i = weights_test.ravel() - weights_test.ravel().mean()
            beta = np.sum(a_i * b_i) / np.sum(b_i**2)
        
        score = weighted_loss - beta * (weights_test.ravel() - 1)
        scores[0].append(score.mean())
    
    return scores

def target_cross_val_scores(model, xsource, ysource, xtarget, ytarget, weights, 
                            metrics=[mean_squared_error], method='lssvr', cv=None):
    """
    Perform target cross-validation for domain adaptation.

    Args:
        model: The model to evaluate.
        xsource (np.ndarray): Source domain feature matrix.
        ysource (np.ndarray): Source domain target vector.
        xtarget (np.ndarray): Target domain feature matrix.
        ytarget (np.ndarray): Target domain target vector.
        weights (np.ndarray): Sample weights.
        metrics (list): List of metric functions to evaluate.
        method (str): Method for fitting the model ('lssvr' or 'svr').
        cv (int): Number of folds for cross-validation.

    Returns:
        list: List of scores for each metric.
    """
    n_s = xsource.shape[0]
    loo = LeaveOneOut() if cv is None else KFold(n_splits=cv)
    scores = [[] for _ in metrics]
    weights_source = weights[:n_s, :]
    weights_target = weights[n_s:, :]

    for train_index, test_index in loo.split(xtarget):
        model_clone = clone(model)
        X_target_train, X_target_test = xtarget[train_index], xtarget[test_index]
        y_target_train, y_target_test = ytarget[train_index], ytarget[test_index]
        weights_target_train, weights_target_test = weights_target[train_index], weights_target[test_index]
        
        X_train = np.vstack((xsource, X_target_train))
        y_train = np.vstack((ysource, y_target_train))
        weights_train = np.vstack((weights_source, weights_target_train))

        if method.lower() == 'lssvr':
            model_clone.fix_fit(X_train, y_train, sample_weight=weights_train)
            y_target_pred = model_clone.fix_predict(X_target_test)
        elif method.lower() == 'svr':
            model_clone.fit(X_train, y_train.ravel(), sample_weight=weights_train.ravel())
            y_target_pred = model_clone.predict(X_target_test)
        else:
            raise ValueError('Unknown method')

        for i, metric in enumerate(metrics):
            score = metric(y_target_test, y_target_pred, sample_weight=weights_target_test.ravel())
            scores[i].append(score)

    return scores
def lssvrtarget_cross_val_scores(model, xtarget, ytarget, metrics=[accuracy_score], cv=None):
    """Perform LSSVR target cross-validation."""
    loo = LeaveOneOut() if cv is None else KFold(n_splits=cv)
    scores = [[] for _ in metrics]

    for train_index, test_index in loo.split(xtarget):
        model_clone = clone(model)
        X_target_train, X_target_test = xtarget[train_index], xtarget[test_index]
        y_target_train, y_target_test = ytarget[train_index], ytarget[test_index]
        model_clone.fit(X_target_train, y_target_train)
        y_target_pred = model_clone.predict(X_target_test)
        for i, metric in enumerate(metrics):
            score = metric(y_target_test, y_target_pred)
            scores[i].append(score)
    return scores
def lssvrcross_val_scores_weighted(model, X, y, weights, cv=5, metrics=[accuracy_score]):
    """Perform weighted cross-validation for LSSVR."""
    kf = KFold(n_splits=cv)
    scores = [[] for _ in metrics]
    for train_index, test_index in kf.split(X):
        model_clone = clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fix_fit(X_train, y_train.ravel(), sample_weight=weights_train.ravel())
        y_pred = model_clone.fix_predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred, sample_weight=weights_test.ravel())
            scores[i].append(score)
    return scores
def lssvrcross_val_scores_controlweighted(model, X, y, weights, cv=5):
    """Perform control-weighted cross-validation for LSSVR."""
    loo = LeaveOneOut() if cv is None else KFold(n_splits=cv)
    scores = [[]]
    for train_index, test_index in loo.split(X):
        model_clone = clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fix_fit(X_train, y_train.ravel(), sample_weight=weights_train.ravel())
        y_pred = model_clone.fix_predict(X_test)
        weighted_loss = (y_pred.ravel() - y_test.ravel())**2 * weights_test.ravel()
        if weights_test.ravel().mean() == 1:
            beta = 0
        else:
            a_i = weighted_loss - weighted_loss.mean()
            b_i = weights_test.ravel() - weights_test.ravel().mean()
            beta = np.sum(a_i * b_i) / np.sum(b_i**2)
        score = weighted_loss - beta * (weights_test.ravel() - 1)
        scores[0].append(score.mean())
    return scores

# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    X = np.array([[1], [2], [1]] * 30)
    y = np.array([0, 1, 1] * 30)
    weights = np.array([98, 1, 1] * 30)
    no_weights = np.array([1, 1, 1] * 30)

    logistic_model = LogisticRegression()
    weighted_scores = cross_val_scores_weighted(logistic_model, X, y, weights, cv=5, metrics=[mean_squared_error])
    no_weighted_scores = cross_val_scores_weighted(logistic_model, X, y, no_weights, cv=5)

    print("Weighted scores:", weighted_scores)
    print("Unweighted scores:", no_weighted_scores)