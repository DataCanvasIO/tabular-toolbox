# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from . import BaseEstimator
from sklearn.metrics import get_scorer

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import GeneralExperiment, ConsoleCallback
from hypernets.searchers import RandomSearcher
from hypernets.core import EarlyStoppingCallback


class HyperGBMEstimator(BaseEstimator):
    def __init__(self, task, scorer, mode='one-stage', max_trails=30, use_cache=True, earlystop_rounds=30, **kwargs):
        super(HyperGBMEstimator, self).__init__(task)
        self.name = 'HyperGBM'
        self.scorer = scorer
        self.mode = mode
        self.kwargs = kwargs
        self.estimator = None
        self.max_trails = max_trails
        self.use_cache = use_cache
        self.earlystop_rounds = earlystop_rounds

    def train(self, X, y, X_test):
        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
                            optimize_direction=OptimizeDirection.Maximize)
        es = EarlyStoppingCallback(self.earlystop_rounds, 'max')
        hk = HyperGBM(rs, reward_metric='auc', cache_dir=f'hypergbm_cache', callbacks=[es])

        log_callback = ConsoleCallback()
        experiment = CompeteExperiment(hk, X, y, X_test=X_test,
                                       callbacks=[log_callback],
                                       scorer=get_scorer(self.scorer),
                                       drop_feature_with_collinearity=False,
                                       drift_detection=True,
                                       mode=self.mode,
                                       n_est_feature_importance=5,
                                       importance_threshold=1e-5,
                                       ensemble_size=10
                                       )
        self.estimator = experiment.run(use_cache=self.use_cache, max_trails=self.max_trails)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)
