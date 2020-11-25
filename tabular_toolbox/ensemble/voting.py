# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .base_ensemble import BaseEnsemble
import numpy as np


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
        if self.task == 'multiclass' and self.method=='hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        proba = np.mean(predictions, axis=1)
        if self.task == 'regression':
            return proba
        proba = np.clip(proba, 0, 1)
        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba
