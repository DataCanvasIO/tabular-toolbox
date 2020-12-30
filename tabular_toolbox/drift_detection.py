# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy
import time

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .column_selector import column_object_category_bool, column_number_exclude_timedelta
from .dataframe_mapper import DataFrameMapper
from .utils import logging
from .sklearn_ex import SafeOrdinalEncoder

logger = logging.getLogger(__name__)

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


def general_preprocessor():
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', SafeOrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    preprocessor = DataFrameMapper(features=[(column_object_category_bool, cat_transformer),
                                             (column_number_exclude_timedelta, num_transformer)],
                                   input_df=True,
                                   df_out=True)
    return preprocessor


class FeatureSelectionCallback():
    def on_round_start(self, round_no, features, ):
        pass

    def on_round_end(self, round_no, auc, features, remove_features, elapsed):
        pass

    def on_remove_shift_variable(self, shift_score, remove_features):
        pass

    def on_task_break(self, round_no, auc, features):
        pass

    def on_task_finished(self, round_no, auc, features):
        pass


def feature_selection(X_train, X_test,
                      remove_shift_variable=True,
                      variable_shift_threshold=0.7,
                      variable_shift_scorer=None,
                      auc_threshold=0.55,
                      min_features=10, remove_size=0.1,
                      preprocessor=None,
                      estimator=None, sample_balance=True, max_test_samples=None, cv=5, random_state=9527,
                      copy_data=False,
                      callbacks=None):
    logger.info('Feature selection to try to eliminate the concept drift.')
    if copy_data:
        X_train = copy.deepcopy(X_train)
        X_test = copy.deepcopy(X_test)
    scores = None
    if remove_shift_variable:
        scores = covariate_shift_score(X_train, X_test, scorer=variable_shift_scorer)
        remain_features = []
        remove_features = []
        for col, score in scores.items():
            if score <= variable_shift_threshold:
                remain_features.append(col)
            else:
                remove_features.append(col)
                logger.info(f'Remove shift varibale:{col},  score:{score}')
        if len(remain_features) < X_train.shape[1]:
            X_train = X_train[remain_features]
            X_test = X_test[remain_features]
        if callbacks is not None:
            for callback in callbacks:
                callback.on_remove_shift_variable(scores, remove_features)
    round = 1
    history = []
    while True:
        start_time = time.time()
        if callbacks is not None:
            for callback in callbacks:
                callback.on_round_start(round_no=round, features=X_train.columns.to_list())
        logger.info(f'\nRound: {round}\n')
        detector = DriftDetector(preprocessor, estimator, random_state)
        detector.fit(X_train, X_test, sample_balance=sample_balance, max_test_samples=max_test_samples, cv=cv)
        logger.info(f'AUC:{detector.auc_}, Features:{detector.feature_names_}')
        elapsed = time.time() - start_time
        history.append({'auc': detector.auc_,
                        'n_features': len(detector.feature_names_),
                        'elapsed': elapsed
                        })

        if detector.auc_ <= auc_threshold:
            logger.info(
                f'AUC:{detector.auc_} has dropped below the threshold:{auc_threshold}, feature selection is over.')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_finished(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        indices = np.argsort(detector.feature_importances_)
        if indices.shape[0] <= min_features:
            logger.warn(f'The number of remaining features is insufficient to continue remove features. '
                        f'AUC:{detector.auc_} '
                        f'Remaining features:{detector.feature_names_}')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        removes = int(indices.shape[0] * remove_size)
        if removes <= 0:
            logger.warn(f'The number of remaining features is insufficient to continue remove features. '
                        f'AUC:{detector.auc_} '
                        f'Remaining features:({len(detector.feature_names_)}) / {detector.feature_names_}')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        if (indices.shape[0] - removes) < min_features:
            removes = indices.shape[0] - min_features

        remain_features = list(np.array(detector.feature_names_)[indices[:-removes]])
        remove_features = list(set(detector.feature_names_) - set(remain_features))
        logger.info(f'Removed features: {remove_features}')
        X_train = X_train[remain_features]
        X_test = X_test[remain_features]
        if callbacks is not None:
            for callback in callbacks:
                callback.on_round_end(round_no=round, auc=detector.auc_, features=detector.feature_names_,
                                      remove_features=remove_features, elapsed=elapsed)
        round += 1


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
        logger.info('Fit data for concept drift detection')
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

        X_train.insert(0, target_col, 0)
        X_test.insert(0, target_col, 1)

        X_merge = pd.concat([X_train, X_test], axis=0)
        y = X_merge.pop(target_col)

        logger.info('Preprocessing...')
        X_merge = self.preprocessor.fit_transform(X_merge)

        self.feature_names_ = X_merge.columns.to_list()
        iterators = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1001)

        self.oof_proba_ = np.zeros((y.shape[0], 1))
        self.feature_importances_ = []
        auc_all = []
        importances = []
        estimators = []

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_merge, y)):
            logger.info(f'Fold:{n_fold + 1}')
            x_train_fold, y_train_fold = X_merge.iloc[train_idx], y.iloc[train_idx]
            x_val_fold, y_val_fold = X_merge.iloc[valid_idx], y.iloc[valid_idx]
            estimator = copy.deepcopy(self.estimator_)
            kwargs = {}
            if isinstance(estimator, LGBMClassifier):
                kwargs['eval_set'] = (x_val_fold, y_val_fold)
                kwargs['early_stopping_rounds'] = 10
                kwargs['verbose'] = 0
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
        X = copy.deepcopy(X)
        assert self.fitted, 'Please fit it first.'

        cat_cols = column_object_category_bool(X)
        num_cols = column_number_exclude_timedelta(X)
        X.loc[:, cat_cols + num_cols] = self.preprocessor.transform(X)
        oof_proba = []
        for i, estimator in enumerate(self.estimator_):
            oof_proba.append(estimator.predict_proba(X)[:, 1])
        proba = np.mean(oof_proba, axis=0)
        return proba

    def train_test_split(self, X, y, test_size=0.25, remain_for_train=0.3):
        proba = self.predict_proba(X)
        sorted_indices = np.argsort(proba)
        target = '__train_test_split_y__'
        X.insert(0, target, y)

        assert remain_for_train < 1.0 and remain_for_train >= 0, '`remain_for_train` must be < 1.0 and >= 0.'
        if isinstance(test_size, float):
            assert test_size < 1.0 and test_size > 0, '`test_size` must be < 1.0 and > 0.'
            test_size = int(X.shape[0] * test_size)
        assert isinstance(test_size, int), '`test_size` can only be int or float'
        split_size = int(test_size + test_size * remain_for_train)
        assert split_size < X.shape[0], \
            'test_size+test_size*remain_for_train must be less than the number of samples in X.'

        if remain_for_train == 0:
            X_train = X.iloc[sorted_indices[:-test_size]]
            X_test = X.iloc[sorted_indices[-test_size:]]
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test
        else:
            X_train_1 = X.iloc[sorted_indices[:-split_size]]
            X_mixed = X.iloc[sorted_indices[-split_size:]]
            X_train_2, X_test = train_test_split(X_mixed, test_size=test_size, shuffle=True,
                                                 random_state=self.random_state)
            X_train = pd.concat([X_train_1, X_train_2], axis=0)
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test


def covariate_shift_score(X_train, X_test, scorer=None, cv=None, copy_data=True):
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test,
                                                            pd.DataFrame), 'X_train and X_test must be a pandas DataFrame.'
    assert len(set(X_train.columns.to_list()) - set(
        X_test.columns.to_list())) == 0, 'The columns in X_train and X_test must be the same.'
    target_col = '__hypernets_csd__target__'
    if scorer is None:
        scorer = roc_auc_scorer
    if copy_data:
        train = copy.deepcopy(X_train)
        test = copy.deepcopy(X_test)
    else:
        train = X_train
        test = X_test

    # Set target value
    train.insert(0, target_col, 0)
    test.insert(0, target_col, 1)
    X_merge = pd.concat([train, test], axis=0)
    y = X_merge.pop(target_col)

    logger.info('Preprocessing...')
    # Preprocess data: imputing and scaling
    preprocessor = general_preprocessor()
    X_merge = preprocessor.fit_transform(X_merge)

    # Calculate the shift score for each column separately.
    scores = {}
    logger.info('Scoring...')
    for c in X_merge.columns:
        x = X_merge[[c]]
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
