# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from tabular_toolbox.ensemble.stacking import StackingEnsemble
from tabular_toolbox.ensemble.voting import AveragingEnsemble
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabular_toolbox.column_selector import column_object_category_bool, column_number_exclude_timedelta
from tabular_toolbox.dataframe_mapper import DataFrameMapper
from tabular_toolbox.datasets import dsutils
from lightgbm import LGBMClassifier


def general_preprocessor():
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', OrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    preprocessor = DataFrameMapper(features=[(column_object_category_bool, cat_transformer),
                                             (column_number_exclude_timedelta, num_transformer)],
                                   input_df=True,
                                   df_out=True)
    return preprocessor


class TestStacking():
    binary_y_true = np.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ])
    bianry_y_preds = np.array([
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0,
         1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, ],
        [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, ],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, ],
    ]).T

    multiclass_y_true = np.array(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
         2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, ])

    multiclass_y_preds = np.array([
        [1, 2, 1, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 3, 1, 2, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
         2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, ],
        [1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
         2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 2, 1, ],
        [1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 2, 3, 1, 2, 2, 1, 2, 3, 1, 2, 3, 1,
         2, 2, 1, 2, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, ],
    ]).T

    def test_multi_estimators(self):
        preprocessor = general_preprocessor()
        df = dsutils.load_bank().head(2000)
        y = df.pop('y')
        X = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=9527)

        estimators = [RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), LogisticRegression(),
                      GradientBoostingClassifier()]

        rf_auc = self.get_auc(estimators[0], X_train, X_test, y_train, y_test)
        dt_auc = self.get_auc(estimators[1], X_train, X_test, y_train, y_test)
        et_auc = self.get_auc(estimators[2], X_train, X_test, y_train, y_test)
        lr_auc = self.get_auc(estimators[3], X_train, X_test, y_train, y_test)
        gb_auc = self.get_auc(estimators[4], X_train, X_test, y_train, y_test)

        ests = [RandomForestClassifier(), LogisticRegression(), GradientBoostingClassifier()]
        avg = AveragingEnsemble('binary', ests, need_fit=True, n_folds=5, )

        avg_auc = self.get_auc(avg, X_train, X_test, y_train, y_test)

        avg_hard = AveragingEnsemble('binary', ests, need_fit=True, n_folds=5, method='hard')
        avg_auc_hard = self.get_auc(avg_hard, X_train, X_test, y_train, y_test)

        ensembler_hard = StackingEnsemble('binary', ests, need_fit=True, n_folds=5, method='hard')
        ensemble_auc_hard = self.get_auc(ensembler_hard, X_train, X_test, y_train, y_test)

        ensembler_soft = StackingEnsemble('binary', ests, need_fit=True, n_folds=5, method='soft')
        ensemble_auc_soft = self.get_auc(ensembler_soft, X_train, X_test, y_train, y_test)
        assert ensemble_auc_soft

    def get_auc(self, clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:, -1]
        auc = roc_auc_score(y_test, pred)
        return auc

    def test_binary(self):
        ensembler = StackingEnsemble('binary', [1, 2, 3])
        ensembler.fit(X=None, y=self.binary_y_true, est_predictions=self.bianry_y_preds)
        assert ensembler.meta_model
        en_pred = ensembler.predict_predictions(self.bianry_y_preds)
        np.testing.assert_equal(self.binary_y_true, en_pred)

        ensembler = StackingEnsemble('binary', [1, 2, 3])
        ensembler.fit_predictions(self.bianry_y_preds, self.binary_y_true)
        en_pred = ensembler.predict_predictions(self.bianry_y_preds)
        np.testing.assert_equal(self.binary_y_true, en_pred)

    def test_multiclass(self):
        ensembler = StackingEnsemble('multiclass')
        ensembler.fit(self.multiclass_y_preds, self.multiclass_y_true)
        assert ensembler.meta_model
        en_pred = ensembler.predict(self.multiclass_y_preds)
        assert en_pred.shape == (54,)

    def test_regression(self):
        ensembler = StackingEnsemble('regression')
        ensembler.fit(self.bianry_y_preds, self.binary_y_true)
        assert ensembler.meta_model
        en_pred = ensembler.predict(self.bianry_y_preds)
        assert en_pred.shape == (60,)
