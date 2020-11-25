# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .base_ensemble import BaseEnsemble
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np


class StackingEnsemble(BaseEnsemble):
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', meta_model=None, fit_kwargs=None):
        super(StackingEnsemble, self).__init__(task, estimators, need_fit, n_folds, method)
        if meta_model is None:
            if task == 'regression':
                self.meta_model = LGBMRegressor(
                    n_estimators=50,
                    num_leaves=15,
                    max_depth=5,
                    subsample=0.5,
                    subsample_freq=0,
                    colsample_bytree=0.8,
                    reg_alpha=1,
                    reg_lambda=1)
            else:
                self.meta_model = LGBMClassifier(
                    n_estimators=50,
                    num_leaves=15,
                    max_depth=5,
                    subsample=0.5,
                    subsample_freq=0,
                    colsample_bytree=0.8,
                    reg_alpha=1,
                    reg_lambda=1,
                )
        else:
            self.meta_model = meta_model
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    def fit_predictions(self, predictions, y_true):
        self.meta_model.fit(predictions, y_true, **self.fit_kwargs)

    def predictions2predict(self, predictions):
        assert self.meta_model is not None
        pred = self.meta_model.predict(predictions)
        if self.task == 'binary':
            pred = np.clip(pred, 0, 1)
        return pred

    def predictions2predict_proba(self, predictions):
        assert self.meta_model is not None
        if hasattr(self.meta_model, 'predict_proba'):
            pred = self.meta_model.predict_proba(predictions)
        else:
            pred = self.meta_model.predict(predictions)

        if self.task == 'binary':
            pred = np.clip(pred, 0, 1)
        return pred
