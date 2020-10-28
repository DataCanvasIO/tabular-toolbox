# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import patsy
import numpy as np
from tabular_toolbox.datasets import dsutils

class Test_Categorical_Feature():
    def test_cat_interaction(self):

        df = dsutils.load_bank().head(100)

        # create dummy variables, and their interactions
        y, X = patsy.dmatrices('age ~ C(job)*C(education)', df, return_type="dataframe")
        # flatten y into a 1-D array so scikit-learn can understand it
        y = np.ravel(y)
        assert y