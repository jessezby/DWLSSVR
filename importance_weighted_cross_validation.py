from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE

from torch import vstack
import sklearn.linear_model
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score


def cross_val_scores_weighted(model,
                              X,
                              y,
                              weights,
                              cv=5,
                              metrics=[sklearn.metrics.accuracy_score]):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = [[] for metric in metrics]
    for train_index, test_index in kf.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fit(X_train,
                        y_train.ravel(),
                        sample_weight=weights_train.ravel())
        y_pred = model_clone.predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred, sample_weight=weights_test.ravel())
            scores[i].append(score)
    return scores


def cross_val_scores_basic(model,
                           X,
                           y,
                           cv=5,
                           metrics=[sklearn.metrics.accuracy_score]):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = [[] for metric in metrics]
    for train_index, test_index in kf.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_clone.fit(X_train, y_train.ravel())
        y_pred = model_clone.predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred)
            scores[i].append(score)
    return scores


def cross_val_scores_controlweighted(model, X, y, weights, cv=5):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = [[] for a in [0]]
    for train_index, test_index in kf.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fit(X_train,
                        y_train.ravel(),
                        sample_weight=weights_train.ravel())
        y_pred = model_clone.predict(X_test)
        weighted_loss = (y_pred.ravel() -
                         y_test.ravel())**2 * weights_test.ravel()
        #score = metric(y_test, y_pred)
        if weights_test.ravel().mean() == 1:
            beta = 0
        else:
            a_i = np.array(weighted_loss) - np.array(weighted_loss).mean()
            b_i = weights_test.ravel() - weights_test.ravel().mean()
            beta = np.sum(a_i * b_i) / np.sum(b_i**2)
        score = (y_pred.ravel() - y_test.ravel())**2 * weights_test.ravel(
        ) - beta * (weights_test.ravel() - 1)
        scores[0].append(score.mean())
    return scores


def cross_val_scores_testweighted(model,
                                  X,
                                  y,
                                  weights,
                                  cv=5,
                                  metrics=[sklearn.metrics.accuracy_score]):
    if cv == None:
        loo = LeaveOneOut()
    else:
        loo = KFold(n_splits=cv)
    loo.get_n_splits(X)
    scores = [[] for metric in metrics]
    for train_index, test_index in loo.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fit(X_train,
                        y_train.ravel(),
                        sample_weight=weights_train.ravel())
        y_pred = model_clone.predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred)
            scores[i].append(score)
    return scores


def target_cross_val_scores(model,
                            xsource,
                            ysource,
                            xtarget,
                            ytarget,
                            weights,
                            metrics=[sklearn.metrics.accuracy_score],
                            method='lssvr',
                            cv=None):
    n_t = xtarget.shape[0]
    n_s = xsource.shape[0]
    if cv == None:
        loo = LeaveOneOut()
    else:
        loo = KFold(n_splits=cv)
    loo.get_n_splits(xtarget)
    scores = [[] for metric in metrics]
    weights_source = weights[:n_s, :]
    weights_target = weights[n_s:, :]

    for train_index, test_index in loo.split(xtarget):
        model_clone = sklearn.base.clone(model)
        X_target_train, X_target_test = xtarget[train_index], xtarget[
            test_index]
        y_target_train, y_target_test = ytarget[train_index], ytarget[
            test_index]
        weights_target_train, weights_target_test = weights_target[
            train_index], weights_target[test_index]
        X_train = np.vstack((xsource, X_target_train))
        y_train = np.vstack((ysource, y_target_train))
        weights_train = np.vstack((weights_source, weights_target_train))
        if method == 'lssvr' or method == 'LSSVR':
            model_clone.fix_fit(X_train, y_train, sample_weight=weights_train)
            y_target_pred = model_clone.fix_predict(X_target_test)
            for i, metric in enumerate(metrics):
                score = metric(y_target_test,
                               y_target_pred,
                               sample_weight=weights_target_test.ravel())
            #score = metric(y_test, y_pred)
            scores[i].append(score)
        elif method == 'svr' or method == 'SVR':
            model_clone.fit(X_train,
                            y_train.ravel(),
                            sample_weight=weights_train.ravel())
            y_target_pred = model_clone.predict(X_target_test)
            for i, metric in enumerate(metrics):
                score = metric(y_target_test,
                               y_target_pred,
                               sample_weight=weights_target_test.ravel())
                scores[i].append(score)
        else:
            raise ValueError('unknown method')
    return scores


def lssvrtarget_cross_val_scores(model,
                                 xtarget,
                                 ytarget,
                                 metrics=[sklearn.metrics.accuracy_score],
                                 cv=None):
    if cv == None:
        loo = LeaveOneOut()
    else:
        loo = KFold(n_splits=cv)
    loo.get_n_splits(xtarget)
    scores = [[] for metric in metrics]

    for train_index, test_index in loo.split(xtarget):
        model_clone = sklearn.base.clone(model)
        X_target_train, X_target_test = xtarget[train_index], xtarget[
            test_index]
        y_target_train, y_target_test = ytarget[train_index], ytarget[
            test_index]
        model_clone.fit(X_target_train, y_target_train)
        y_target_pred = model_clone.predict(X_target_test)
        for i, metric in enumerate(metrics):
            score = metric(y_target_test, y_target_pred)
            #score = metric(y_test, y_pred)
            scores[i].append(score)
    return scores


def lssvrcross_val_scores_weighted(model,
                                   X,
                                   y,
                                   weights,
                                   cv=5,
                                   metrics=[sklearn.metrics.accuracy_score]):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = [[] for metric in metrics]
    for train_index, test_index in kf.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fix_fit(X_train,
                            y_train.ravel(),
                            sample_weight=weights_train.ravel())
        y_pred = model_clone.fix_predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred, sample_weight=weights_test.ravel())
            scores[i].append(score)
    return scores


def lssvrcross_val_scores_controlweighted(model, X, y, weights, cv=5):
    if cv == None:
        loo = LeaveOneOut()
    else:
        loo = KFold(n_splits=cv)
    loo.get_n_splits(X)
    scores = [[] for a in [0]]
    for train_index, test_index in loo.split(X):
        model_clone = sklearn.base.clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fix_fit(X_train,
                            y_train.ravel(),
                            sample_weight=weights_train.ravel())
        y_pred = model_clone.fix_predict(X_test)
        weighted_loss = (y_pred.ravel() -
                         y_test.ravel())**2 * weights_test.ravel()
        if weights_test.ravel().mean() == 1:
            beta = 0
        else:
            a_i = np.array(weighted_loss) - np.array(weighted_loss).mean()
            b_i = weights_test.ravel() - weights_test.ravel().mean()
            beta = np.sum(a_i * b_i) / np.sum(b_i**2)
        score = (y_pred.ravel() - y_test.ravel())**2 * weights_test.ravel(
        ) - beta * (weights_test.ravel() - 1)
        scores[0].append(score.mean())
    return scores


X = np.array([[1], [2], [1]] * 30)
y = np.array([0, 1, 1] * 30)
weights = np.array([98, 1, 1] * 30)
no_weights = np.array([1, 1, 1] * 30)
rng = np.random.RandomState(1)
X1 = rng.normal(loc=0.5, scale=0.5, size=(100, 1))
Y1 = -X1 + np.power(X1, 3)
Y1 += rng.normal(0, 0.3, size=Y1.shape)

rng = np.random.RandomState(2)
Xp = rng.normal(loc=0., scale=0.3, size=(20, 1))
Yp = -Xp + np.power(Xp, 3) + 1
Yp += rng.normal(0, 0.3, size=Yp.shape)

logistic_model = sklearn.linear_model.LogisticRegression()
weighted_scores = cross_val_scores_weighted(logistic_model,
                                            X,
                                            y,
                                            weights,
                                            cv=5,
                                            metrics=[mean_squared_error])
no_weighted_scores = cross_val_scores_weighted(logistic_model,
                                               X,
                                               y,
                                               no_weights,
                                               cv=5)

print(weighted_scores, no_weighted_scores)