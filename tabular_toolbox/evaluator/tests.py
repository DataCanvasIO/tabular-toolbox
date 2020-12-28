# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .hypergbm import HyperGBMEstimator
from .auto_sklearn import AutoSklearnEstimator
from .h2o import H2OEstimator
from .hyperdt import HyperDTEstimator
from . import Evaluator
from ..datasets import dsutils
from hypergbm.search_space import search_space_feature_gen


class Test_Evaluator():

    def test_all(self):
        X = dsutils.load_glass_uci()
        hypergbm_estimator = HyperGBMEstimator(task='multiclass', scorer='roc_auc_ovo')
        autosklearn_estimator = AutoSklearnEstimator(task='multiclass', time_left_for_this_task=30,
                                                     per_run_time_limit=10)
        h2o_estimator = H2OEstimator(task='multiclass')
        hyperdt_estimator = HyperDTEstimator(task='multiclass', reward_metric='AUC', max_trails=3, epochs=1)
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target=10,
                                    task='multiclas',
                                    estimators=[
                                        # autosklearn_estimator,
                                        # hypergbm_estimator,
                                        # h2o_estimator,
                                        hyperdt_estimator,
                                    ],
                                    scorers=['accuracy', 'roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result

    def test_all_binary(self):
        X = dsutils.load_blood() #.load_bank().head(1000)
        task = 'binary'
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo')
        hypergbm_estimator_fg = HyperGBMEstimator(task=task, scorer='roc_auc_ovo', max_trails=3,
                                                  search_space_fn=lambda: search_space_feature_gen(
                                                      early_stopping_rounds=20, verbose=0, task=task))

        autosklearn_estimator = AutoSklearnEstimator(task=task, time_left_for_this_task=30,
                                                     per_run_time_limit=10)
        h2o_estimator = H2OEstimator(task=task)
        hyperdt_estimator = HyperDTEstimator(task=task, reward_metric='AUC', max_trails=3, epochs=1)
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target='Class',
                                    task=task,
                                    estimators=[
                                        # autosklearn_estimator,
                                        # hypergbm_estimator,
                                        # h2o_estimator,
                                        hyperdt_estimator,
                                        #hypergbm_estimator_fg
                                    ],
                                    scorers=['accuracy', 'roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result
