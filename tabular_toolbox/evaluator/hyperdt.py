# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from deeptables.models.hyper_dt import HyperDT
from deeptables.models.hyper_dt import mini_dt_space
from hypernets.core import EarlyStoppingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers import EvolutionSearcher
from sklearn.model_selection import train_test_split

from . import BaseEstimator


class HyperDTEstimator(BaseEstimator):
    def __init__(self, task, reward_metric, max_trails=30, epochs=100, earlystop_rounds=30, **kwargs):
        super(HyperDTEstimator, self).__init__(task)
        self.name = 'HyperDT'
        self.kwargs = kwargs
        self.estimator = None
        self.max_trails = max_trails
        self.reward_metric = reward_metric
        self.epochs = epochs
        self.earlystop_rounds = earlystop_rounds

    def train(self, X, y, X_test):
        searcher = EvolutionSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize, population_size=30,
                                     sample_size=10, regularized=True, candidates_size=10)
        es = EarlyStoppingCallback(self.earlystop_rounds, 'max')

        hdt = HyperDT(searcher,
                      callbacks=[es],
                      reward_metric=self.reward_metric,
                      cache_preprocessed_data=True,
                      )
        stratify = y
        if self.task == 'regression':
            stratify = None
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3,
                                                            random_state=9527, stratify=stratify)

        hdt.search(X_train, y_train, X_eval, y_eval, max_trails=self.max_trails, epochs=self.epochs)
        best_trial = hdt.get_best_trail()
        self.estimator = hdt.load_estimator(best_trial.model_file)

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        if self.task == 'binary':
            proba = np.hstack((1 - proba, proba))
        return proba

    def predict(self, X):
        return self.estimator.predict(X)
