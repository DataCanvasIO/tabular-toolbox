# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .hypergbm import HyperGBMEstimator
from .auto_sklearn import AutoSklearnEstimator
from .h2o import H2OEstimator
from . import Evaluator


class Test_Evaluator():

    def test_all(self):
        from pycaret.datasets import get_data
        X = get_data('glass')
        hypergbm_estimator = HyperGBMEstimator(task='multiclass', scorer='roc_auc_ovo')
        autosklearn_estimator = AutoSklearnEstimator(task='multiclass', time_left_for_this_task=30,
                                                     per_run_time_limit=10)
        h2o_estimator = H2OEstimator(task='multiclass')
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target='Type',
                                    task='multiclas',
                                    estimators=[
                                        autosklearn_estimator,
                                        hypergbm_estimator,
                                        h2o_estimator,
                                    ],
                                    scorers=['accuracy', 'roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result
