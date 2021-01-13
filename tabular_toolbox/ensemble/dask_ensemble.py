# -*- coding:utf-8 -*-
"""

"""

from collections import defaultdict

import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask_ml.model_selection import KFold
from hypernets.utils import logging
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _PredictScorer
from tabular_toolbox import dask_ex as dex

from .base_ensemble import BaseEnsemble

logger = logging.get_logger(__name__)


class DaskGreedyEnsemble(BaseEnsemble):

    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527,
                 scoring='neg_log_loss', ensemble_size=0):
        super().__init__(task, estimators, need_fit, n_folds, method, random_state)

        self.scorer = get_scorer(scoring)
        self.ensemble_size = ensemble_size

        # fitted
        self.weights_ = None
        self.scores_ = None
        self.best_stack_ = None
        self.hits_ = None

    # def __predict(self, estimator, X):
    #     if self.task == 'regression':
    #         pred = estimator.predict(X)
    #     else:
    #         if self.classes_ is None and hasattr(estimator, 'classes_'):
    #             self.classes_ = estimator.classes_
    #         pred = estimator.predict_proba(X)
    #         if self.method == 'hard':
    #             pred = self.proba2predict(pred)
    #     return pred

    def X2predictions(self, X):
        preds = [self.__predict(est, X) for est in self.estimators]
        return dex.hstack_array(preds)

    def X2predictions_by_fit(self, X, y):
        raise NotImplementedError('fixme')

        preds = []
        est_predictions = None

        columns = X.columns
        X_values, y_values = X.to_dask_array(lengths=True), y.to_dask_array(lengths=True)
        iterators = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for fold, (train, test) in enumerate(iterators.split(X_values, y_values)):
            X_train, y_train = X_values[train], y_values[train]
            X_test, y_test = X_values[test], y_values[test]
            X_train = dd.from_dask_array(X_train, columns=columns)
            x_test = dd.from_dask_array(X_test, columns=columns)

            for n, estimator in enumerate(self.estimators):
                estimator.fit(X_train, y_train)
                pred = self.__predict(estimator, X_test)
                if est_predictions is None:
                    est_predictions = np.zeros((len(y), len(self.estimators), pred.shape[1]), dtype=np.float64)
                est_predictions[test, n] = pred

        return dex.hstack_array(preds)

    def fit(self, X, y_true, predictions=None):
        assert any(t is not None for t in [X, predictions])
        assert predictions is None or isinstance(predictions, (tuple, list))
        assert y_true is not None

        def get_prediction(j):
            if predictions is not None and predictions[j] is not None:
                pred = predictions[j]
                if self.method == 'hard' and len(pred.shape) > 1 and pred.shape[1] > 1:
                    pred = self.proba2predict(pred)
            elif X is not None:
                pred = self.__predict(self.estimators[j], X)
            else:
                raise ValueError(f'Not found predictions for esitimator {j}')
            return pred

        if len(self.estimators) == 1:
            self.weights_ = [1]
            return self

        scores = []
        best_stack = []
        sum_predictions = None
        n_estimators = len(self.estimators)
        size = self.ensemble_size if self.ensemble_size > 0 else n_estimators

        logger.info(f'start ensemble {n_estimators} estimators')

        if dex.is_dask_object(y_true):
            y_true = y_true.compute()

        for i in range(size):
            stack_scores = []
            stack_preds = []
            for j in range(n_estimators):
                pred = get_prediction(j)
                if sum_predictions is None:
                    sum_predictions = da.zeros_like(pred, dtype=np.float64)
                mean_predictions = (sum_predictions + pred) / (len(best_stack) + 1)
                if isinstance(self.scorer, _PredictScorer):
                    if self.classes_ is not None:
                        # pred = np.array(self.classes_).take(np.argmax(mean_predictions, axis=1), axis=0)
                        pred = da.take(da.array(self.classes_), da.argmax(mean_predictions, axis=1), axis=0)
                    mean_predictions = pred
                elif self.task == 'binary' and len(mean_predictions.shape) == 2 and mean_predictions.shape[1] == 2:
                    mean_predictions = mean_predictions[:, 1]
                if dex.is_dask_object(mean_predictions):
                    mean_predictions = mean_predictions.compute()
                score = self.scorer._score_func(y_true, mean_predictions, **self.scorer._kwargs) * self.scorer._sign
                stack_scores.append(score)
                stack_preds.append(pred)

            best = np.argmax(stack_scores)
            scores.append(stack_scores[best])
            best_stack.append(best)
            sum_predictions += stack_preds[best]

        # sum up estimator's hit count
        val_steps = len(best_stack)
        hits = defaultdict(int)
        for i in range(val_steps):
            hits[best_stack[i]] += 1

        weights = np.zeros((len(self.estimators)), dtype=np.float64)
        for i in range(len(self.estimators)):
            if hits.get(i) is not None:
                weights[i] = hits[i] / val_steps

        estimators_ = []
        self.weights_ = []
        for index in np.argwhere(weights > 0.).ravel():
            estimators_.append(self.estimators[index])
            self.weights_.append(weights[index])

        self.estimators = estimators_
        self.scores_ = scores
        self.hits_ = hits
        self.best_stack_ = best_stack

        logger.info(f'ensembled with {len(self.estimators)} estimators, weight:{self.weights_}')
        return self

    def predict(self, X):
        if len(self.estimators) == 1:
            preds = self.estimators[0].predict(X)
        else:
            preds = self.proba2predict(self.predict_proba(X))
            if self.task != 'regression' and self.classes_ is not None:
                preds = da.take(da.array(self.classes_), preds, axis=0)

        return preds

    def predict_proba(self, X):
        if len(self.estimators) == 1:
            proba = self.estimators[0].predict_proba(X)
        else:
            proba = None
            for estimator, weight in zip(self.estimators, self.weights_):
                est_proba = estimator.predict_proba(X) * weight
                if proba is None:
                    proba = est_proba
                else:
                    proba += est_proba

        return proba