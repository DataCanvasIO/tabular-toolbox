# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from collections import defaultdict
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _PredictScorer

from .base_ensemble import BaseEnsemble


class AveragingEnsemble(BaseEnsemble):
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft'):
        super(AveragingEnsemble, self).__init__(task, estimators, need_fit, n_folds, method)

    def fit_predictions(self, predictions, y_true):
        return self

    def predictions2predict(self, predictions):
        if len(predictions.shape) == 3 and self.task == 'binary':
            predictions = predictions[:, :, -1]
        proba = np.mean(predictions, axis=1)
        pred = self.proba2predict(proba)
        return pred

    def predictions2predict_proba(self, predictions):
        if self.task == 'multiclass' and self.method == 'hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        proba = np.mean(predictions, axis=1)
        if self.task == 'regression':
            return proba
        proba = np.clip(proba, 0, 1)
        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba


class GreedyEnsemble(BaseEnsemble):
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', scoring='neg_log_loss',
                 ensemble_size=0):
        super(GreedyEnsemble, self).__init__(task, estimators, need_fit, n_folds, method)
        self.weights_ = None
        self.scores_ = None
        self.best_stack_ = None
        self.hits_ = None
        self.scorer = get_scorer(scoring)
        self.ensemble_size = ensemble_size

    def fit_predictions(self, predictions, y_true):
        scores = []
        best_stack = []
        hits = defaultdict(int)
        if len(predictions.shape) == 1:
            self.weights_ = [1]
            return
        elif len(predictions.shape) == 2:
            sum_predictions = np.zeros((predictions.shape[0]), dtype=np.float64)
        elif len(predictions.shape) == 3:
            sum_predictions = np.zeros((predictions.shape[0], predictions.shape[2]), dtype=np.float64)
        else:
            raise ValueError(f'Wrong shape of predictions. shape:{predictions.shape}')

        if self.ensemble_size <= 0:
            size = predictions.shape[1]
        else:
            size = self.ensemble_size
        for i in range(size):
            stack_scores = []
            for j in range(predictions.shape[1]):
                pred = predictions[:, j, :]
                mean_predictions = (sum_predictions + pred) / (len(best_stack) + 1)
                if isinstance(self.scorer, _PredictScorer):
                    pred = np.array(self.classes_).take(np.argmax(mean_predictions, axis=1), axis=0)
                    mean_predictions = pred
                elif self.task == 'binary' and len(mean_predictions.shape) == 2 and mean_predictions.shape[1] == 2:
                    mean_predictions = mean_predictions[:, 1]
                score = self.scorer._score_func(y_true, mean_predictions) * self.scorer._sign
                stack_scores.append(score)

            best = np.argmax(stack_scores)
            scores.append(stack_scores[best])
            best_stack.append(best)
            hits[best] += 1
            sum_predictions += predictions[:, best, :]

        self.weights_ = np.zeros((len(self.estimators)), dtype=np.float64)
        for i in range(len(self.estimators)):
            if hits.get(i) is not None:
                self.weights_[i] = hits[i] / len(best_stack)
        self.scores_ = scores
        self.hits_ = hits
        self.best_stack_ = best_stack

    def predictions2predict(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        if len(predictions.shape) == 3 and self.task == 'binary':
            predictions = predictions[:, :, -1]
        proba = np.sum(predictions * self.weights_, axis=1)
        pred = self.proba2predict(proba)
        return pred

    def predictions2predict_proba(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        if self.task == 'multiclass' and self.method == 'hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        if len(predictions.shape) == 3:
            weights = np.expand_dims(self.weights_, axis=1).repeat(predictions.shape[2], 1)
        else:
            weights = self.weights_
        proba = np.sum(predictions * weights, axis=1)
        if self.task == 'regression':
            return proba
        proba = np.clip(proba, 0, 1)
        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba
