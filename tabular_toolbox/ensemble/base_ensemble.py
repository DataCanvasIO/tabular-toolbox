# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


class BaseEnsemble():
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527):
        self.task = task
        self.estimators = estimators
        self.need_fit = need_fit
        self.method = method
        self.n_folds = n_folds
        self.random_state = random_state

    def __predict(self, estimator, X):
        if self.task != 'binary':
            pred = estimator.predict(X)
        else:
            proba = estimator.predict_proba(X)
            if self.method == 'hard':
                pred = self.proba2predict(proba)
            else:
                pred = proba[:, 1]
        return pred

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task != 'binary':
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict

    def fit(self, X, y, est_predictions=None):
        assert y is not None
        if est_predictions is not None:
            assert est_predictions.shape == (len(y), len(self.estimators))
        else:
            assert X is not None
            est_predictions = np.zeros((len(y), len(self.estimators)), dtype=np.float64)
            if self.need_fit:
                iterators = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                for fold, (train, test) in enumerate(iterators.split(X, y)):
                    for n, estimator in enumerate(self.estimators):
                        X_train = X.iloc[train]
                        y_train = y.iloc[train]
                        X_test = X.iloc[test]
                        estimator.fit(X_train, y_train)
                        est_predictions[test, n] = self.__predict(estimator, X_test)
            else:
                for n, estimator in enumerate(self.estimators):
                    est_predictions[:, n] = self.__predict(estimator, X)

        self.fit_predictions(est_predictions, y)

    def predict(self, X):
        est_predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float64)
        for n, estimator in enumerate(self.estimators):
            est_predictions[:, n] = self.__predict(estimator, X)
        return self.predictions2predict(est_predictions)

    def predict_proba(self, X):
        est_predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float64)
        for n, estimator in enumerate(self.estimators):
            est_predictions[:, n] = self.__predict(estimator, X)
        return self.predictions2predict_proba(est_predictions)

    def fit_predictions(self, predictions, y_true):
        raise NotImplementedError()

    def predictions2predict_proba(self, predictions):
        raise NotImplementedError()

    def predictions2predict(self, predictions):
        raise NotImplementedError()
