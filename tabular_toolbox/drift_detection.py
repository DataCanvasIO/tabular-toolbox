# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from copy import deepcopy
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .column_selector import column_object_category_bool, column_number_exclude_timedelta
from .utils import logging
import copy

logger = logging.getLogger(__name__)

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


def general_preprocessor():
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', OrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, column_object_category_bool),
                                                   ('num', num_transformer, column_number_exclude_timedelta)],
                                     remainder='passthrough')
    return preprocessor


def feature_selection(X_train, X_test, remove_shift_variable=True, auc_threshold=0.55, min_features=10, remove_size=0.2,
                      preprocessor=None,
                      estimator=None, sample_balance=True, max_test_samples=None, cv=5, random_state=9527,
                      copy_data=False):
    if copy_data:
        X_train = copy.deepcopy(X_train)
        X_test = copy.deepcopy(X_test)

    if remove_shift_variable:
        scores = covariate_shift_score(X_train, X_test)
        remain_features = []
        for col, score in scores.items():
            if score <= 0.7:
                remain_features.append(col)
        if len(remain_features) < X_train.shape[1]:
            X_train = X_train[remain_features]
            X_test = X_test[remain_features]

    while True:
        detector = DriftDetector(preprocessor, estimator, random_state)
        detector.fit(X_train, X_test, sample_balance=sample_balance, max_test_samples=max_test_samples, cv=cv)

        if detector.auc_ <= auc_threshold:
            return detector.feature_names_

        indices = np.argsort(detector.feature_importances_)
        if indices.shape[0] <= min_features:
            logger.warn(f'The number of remaining features is insufficient to continue remove features. '
                        f'AUC:{detector.auc_} '
                        f'Remaining features:{detector.feature_names_}')
            return detector.feature_names_

        removes = int(indices.shape[0] * remove_size)
        if (indices.shape[0] - removes) < min_features:
            removes = indices.shape[0] - min_features

        remain_features = list(np.array(detector.feature_names_)[indices[:-removes]])
        X_train = X_train[remain_features]
        X_test = X_test[remain_features]


class DriftDetector():
    def __init__(self, preprocessor=None, estimator=None, random_state=9527):
        if preprocessor is None:
            self.preprocessor = general_preprocessor()
        else:
            self.preprocessor = preprocessor

        if estimator is None or preprocessor == 'gbm':
            self.estimator_ = LGBMClassifier(n_estimators=50,
                                             num_leaves=15,
                                             max_depth=5,
                                             subsample=0.5,
                                             subsample_freq=1,
                                             colsample_bytree=0.8,
                                             reg_alpha=1,
                                             reg_lambda=1,
                                             importance_type='gain', )
        elif preprocessor == 'dt':
            self.preprocessor = DecisionTreeClassifier(min_samples_leaf=20, min_impurity_decrease=0.01)
        elif preprocessor == 'rf':
            self.preprocessor = RandomForestClassifier(min_samples_leaf=20, min_impurity_decrease=0.01)
        else:
            self.estimator_ = estimator

        self.random_state = random_state
        self.auc_ = None
        self.oof_proba_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.fitted = False

    def fit(self, X_train, X_test, sample_balance=True, max_test_samples=None, cv=5):
        assert X_train.shape[1] == X_test.shape[1], 'The number of columns in X_train and X_test must be the same.'
        assert len(set(X_train.columns.to_list()) - set(
            X_test.columns.to_list())) == 0, 'The name of columns in X_train and X_test must be the same.'

        if max_test_samples is not None and max_test_samples < X_test.shape[0]:
            X_test, _ = train_test_split(X_test, train_size=max_test_samples, shuffle=True,
                                         random_state=self.random_state)
        if sample_balance:
            if X_test.shape[0] > X_train.shape[0]:
                X_test, _ = train_test_split(X_test, train_size=X_train.shape[0], shuffle=True,
                                             random_state=self.random_state)
            else:
                X_train, _ = train_test_split(X_train, train_size=X_test.shape[0], shuffle=True,
                                              random_state=self.random_state)

        target_col = '__drift_detection_target__'

        X_train[target_col] = 0
        X_test[target_col] = 1

        X_merge = pd.concat([X_train, X_test], axis=0)
        y = X_merge.pop(target_col)

        cat_cols = column_object_category_bool(X_merge)
        num_cols = column_number_exclude_timedelta(X_merge)
        X_merge[cat_cols + num_cols] = self.preprocessor.fit_transform(X_merge)
        self.feature_names_ = X_merge.columns.to_list()
        iterators = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1001)

        self.oof_proba_ = np.zeros((y.shape[0], 1))
        self.feature_importances_ = []
        auc_all = []
        importances = []
        estimators = []

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_merge, y)):
            print(f'\nFold:{n_fold + 1}\n')

            x_train_fold, y_train_fold = X_merge.iloc[train_idx], y.iloc[train_idx]
            x_val_fold, y_val_fold = X_merge.iloc[valid_idx], y.iloc[valid_idx]

            estimator = copy.deepcopy(self.estimator_)
            kwargs = {}
            if isinstance(estimator, LGBMClassifier):
                kwargs['eval_set'] = (x_val_fold, y_val_fold)
                kwargs['early_stopping_rounds'] = 10

            estimator.fit(x_train_fold, y_train_fold, **kwargs)
            proba = estimator.predict_proba(x_val_fold)[:, 1:2]
            self.oof_proba_[valid_idx] = proba
            auc = roc_auc_score(y_val_fold, proba)
            auc_all.append(auc)
            estimators.append(estimator)
            importances.append(estimator.feature_importances_)

        self.estimator_ = estimators
        self.auc_ = np.mean(auc_all)
        self.feature_importances_ = np.mean(importances, axis=0)
        self.fitted = True
        return self

    def predict_proba(self, X):
        assert self.fitted, 'Please fit it first.'

        cat_cols = column_object_category_bool(X)
        num_cols = column_number_exclude_timedelta(X)
        X[cat_cols + num_cols] = self.preprocessor.transform(X)
        oof_proba = []
        for i, estimator in enumerate(self.estimator_):
            oof_proba.append(estimator.predict_proba(X)[:, 1])
        proba = np.mean(oof_proba, axis=0)
        return proba


def covariate_shift_score(X_train, X_test, scorer=roc_auc_scorer, cv=None, copy_data=True):
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test,
                                                            pd.DataFrame), 'X_train and X_test must be a pandas DataFrame.'
    assert len(set(X_train.columns.to_list()) - set(
        X_test.columns.to_list())) == 0, 'The columns in X_train and X_test must be the same.'
    target_col = '__hypernets_csd__target__'
    if copy_data:
        train = deepcopy(X_train)
        test = deepcopy(X_test)
    else:
        train = X_train
        test = X_test

    # Set target value
    train[target_col] = 0
    test[target_col] = 1
    mixed = pd.concat([train, test], axis=0)
    y = mixed.pop(target_col)

    logger.info('Preprocessing...')
    # Preprocess data: imputing and scaling
    cat_cols = column_object_category_bool(mixed)
    num_cols = column_number_exclude_timedelta(mixed)
    preprocessor = general_preprocessor()
    mixed[cat_cols + num_cols] = preprocessor.fit_transform(mixed)

    # Calculate the shift score for each column separately.
    scores = {}
    logger.info('Scoring...')
    for c in mixed.columns:
        x = mixed[[c]]
        model = LGBMClassifier()
        if cv is None:
            mixed_x_train, mixed_x_test, mixed_y_train, mixed_y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=9527, stratify=y)

            model.fit(mixed_x_train, mixed_y_train, eval_set=(mixed_x_test, mixed_y_test), early_stopping_rounds=20,
                      verbose=False)
            score = scorer(model, mixed_x_test, mixed_y_test)
        else:
            score_ = cross_val_score(model, X=x, y=y, verbose=0, scoring=scorer, cv=cv)
            score = np.mean(score_)
        logger.info(f'column:{c}, score:{score}')
        scores[c] = score

    return scores
