# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from . import BaseEstimator
from sklearn.metrics import get_scorer

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general, search_space_feature_gen
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import GeneralExperiment, ConsoleCallback
from hypernets.searchers import RandomSearcher, EvolutionSearcher, MCTSSearcher
from hypernets.core import EarlyStoppingCallback, SummaryCallback


class HyperGBMEstimator(BaseEstimator):
    def __init__(self, task, scorer, mode='one-stage', max_trails=30, use_cache=True, earlystop_rounds=30,
                 search_space_fn=None, ensemble_size=10, use_meta_learner=False, eval_size=0.3, **kwargs):
        super(HyperGBMEstimator, self).__init__(task)
        self.name = 'HyperGBM'
        self.scorer = scorer
        self.mode = mode
        self.kwargs = kwargs
        self.estimator = None
        self.max_trails = max_trails
        self.use_cache = use_cache
        self.earlystop_rounds = earlystop_rounds
        self.search_space_fn = search_space_fn if search_space_fn is not None else lambda: search_space_general(
            early_stopping_rounds=20, verbose=0)
        self.ensemble_size = ensemble_size
        self.experiment = None
        self.use_meta_learner = use_meta_learner
        self.eval_size = eval_size

    def train(self, X, y, X_test):
        # searcher = MCTSSearcher(self.search_space_fn, use_meta_learner=self.use_meta_learner, max_node_space=10,
        #                         candidates_size=10,
        #                         optimize_direction=OptimizeDirection.Maximize)
        searcher = EvolutionSearcher(self.search_space_fn,
                                     optimize_direction=OptimizeDirection.Maximize, population_size=30, sample_size=10,
                                     regularized=True, candidates_size=10, use_meta_learner=self.use_meta_learner)
        # searcher = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
        #                     optimize_direction=OptimizeDirection.Maximize)
        es = EarlyStoppingCallback(self.earlystop_rounds, 'max')

        hk = HyperGBM(searcher, reward_metric='auc', cache_dir=f'hypergbm_cache', clear_cache=False,
                      callbacks=[es, SummaryCallback()])

        log_callback = ConsoleCallback()
        self.experiment = CompeteExperiment(hk, X, y, X_test=X_test, eval_size=self.eval_size,
                                            callbacks=[log_callback],
                                            scorer=get_scorer(self.scorer),
                                            drop_feature_with_collinearity=False,
                                            drift_detection=True,
                                            mode=self.mode,
                                            n_est_feature_importance=5,
                                            importance_threshold=1e-5,
                                            ensemble_size=self.ensemble_size
                                            )
        self.estimator = self.experiment.run(use_cache=self.use_cache, max_trails=self.max_trails)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)
