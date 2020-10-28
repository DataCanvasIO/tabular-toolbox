# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
import copy

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .column_selector import column_skewness_kurtosis, column_object, column_int, column_object_category_bool
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from .utils import logging

logger = logging.get_logger(__name__)


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average', squared=True):
    return np.sqrt(
        mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=squared))


def subsample(X, y, max_samples, train_samples, task, random_state=9527):
    stratify = None
    if X.shape[0] > max_samples:
        if task != 'regression':
            stratify = y
        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=max_samples, shuffle=True, stratify=stratify
        )
        if task != 'regression':
            stratify = y_train

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, train_size=train_samples, shuffle=True, stratify=stratify, random_state=random_state
        )
    else:
        if task != 'regression':
            stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=True, stratify=stratify
        )

    return X_train, X_test, y_train, y_test


class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
        return y


class MultiLabelEncoder:
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        for n in range(n_features):
            le = SafeLabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())
        for n in range(n_features):
            X[:, n] = self.encoders[n].transform(X[:, n])
        return X


class SkewnessKurtosisTransformer:
    def __init__(self, transform_fn=None, skew_threshold=0.5, kurtosis_threshold=0.5):
        self.columns_ = []
        self.skewness_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        if transform_fn is None:
            transform_fn = np.log
        self.transform_fn = transform_fn

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        self.columns_ = column_skewness_kurtosis(X, skew_threshold=self.skewness_threshold,
                                                 kurtosis_threshold=self.kurtosis_threshold)
        logger.info(f'Selected columns:{self.columns_}')
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        if len(self.columns_) > 0:
            try:
                X[self.columns_] = self.transform_fn(X[self.columns_])
            except Exception as e:
                logger.error(e)
        return X


def reduce_mem_usage(df, verbose=True):
    """
    Adaption from :https://blog.csdn.net/xckkcxxck/article/details/88170281
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))


class DataCleaner:
    def __init__(self, nan_chars=None, deduce_object_dtype=True, clean_invalidate_columns=True,
                 clean_label_nan_rows=True, replace_inf_values=np.nan, drop_columns=None, reduce_mem_usage=False,
                 int_convert_to='float'):
        self.nan_chars = nan_chars
        self.deduce_object_dtype = deduce_object_dtype
        self.clean_invalidate_columns = clean_invalidate_columns
        self.clean_label_nan_rows = clean_label_nan_rows
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.df_meta = None
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to

    def clean_data(self, X, y):
        assert isinstance(X, pd.DataFrame)
        if y is not None:
            X.insert(0, 'hypergbm__Y__', y)

        if self.nan_chars is not None:
            logger.info(f'Replace chars{self.nan_chars} to NaN')
            X = X.replace(self.nan_chars, np.nan)

        if self.deduce_object_dtype:
            logger.info('Deduce data type for object columns.')
            for col in column_object(X):
                try:
                    X[col] = X[col].astype('float')
                except Exception as e:
                    logger.error(f'Deduce object column [{col}] failed. {e}')

        if self.int_convert_to is not None:
            logger.info(f'Convert int type to {self.int_convert_to}')
            int_cols = column_int(X)
            X[int_cols] = X[int_cols].astype(self.int_convert_to)

        if y is not None:
            if self.clean_label_nan_rows:
                logger.info('Clean the rows which label is NaN')
                X = X.dropna(subset=['hypergbm__Y__'])
            y = X.pop('hypergbm__Y__')

        if self.drop_columns is not None:
            logger.info(f'Drop columns:{self.drop_columns}')
            for col in self.drop_columns:
                X.pop(col)

        if self.clean_invalidate_columns:
            logger.info('Clean invalidate columns')
            for col in X.columns:
                n_unique = X[col].nunique(dropna=True)
                if n_unique <= 1:
                    X.pop(col)

        o_cols = column_object(X)
        X[o_cols] = X[o_cols].astype('str')
        return X, y

    def fit_transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)

        X, y = self.clean_data(X, y)
        if self.reduce_mem_usage:
            logger.info('Reduce memory usage')
            reduce_mem_usage(X)

        if self.replace_inf_values is not None:
            logger.info(f'Replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        logger.info('Collect meta info from data')
        df_meta = {}
        for col_info in zip(X.columns.to_list(), X.dtypes):
            dtype = str(col_info[1])
            if df_meta.get(dtype) is None:
                df_meta[dtype] = []
            df_meta[dtype].append(col_info[0])
        self.df_meta = df_meta
        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)
        self.clean_data(X, y)
        if self.df_meta is not None:
            logger.info('Processing with meta info')
            all_cols = []
            for dtype, cols in self.df_meta.items():
                all_cols += cols
                X[cols] = X[cols].astype(dtype)
            drop_cols = set(X.columns.to_list()) - set(all_cols)

            for c in drop_cols:
                X.pop(c)
            logger.info(f'droped columns:{drop_cols}')

        if self.replace_inf_values is not None:
            logger.info(f'Replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        return X, y


class FeatureSelectionTransformer():
    def __init__(self, task, max_train_samples=10000, max_test_samples=10000, max_cols=10000, ratio_max_cols=0.05,
                 n_max_cols=60):
        self.task = task
        if max_cols <= 0:
            max_cols = 10000
        if max_train_samples <= 0:
            max_train_samples = 10000
        if max_test_samples <= 0:
            max_test_samples = 10000

        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.max_cols = max_cols
        self.ratio_max_cols = ratio_max_cols
        self.n_max_cols = n_max_cols
        self.scores_ = {}
        self.columns_ = []

    def get_categorical_features(self, X):
        cat_cols = column_object_category_bool(X)
        int_cols = column_int(X)
        for c in int_cols:
            if X[c].min() >= 0 and X[c].max() < np.iinfo(np.int32).max:
                cat_cols.append(c)
        return cat_cols

    def feature_score(self, F_train, y_train, F_test, y_test):
        if self.task == 'regression':
            model = LGBMRegressor()
            eval_metric = root_mean_squared_error
        else:
            model = LGBMClassifier()
            eval_metric = log_loss

        cat_cols = self.get_categorical_features(F_train)

        model.fit(F_train, y_train,
                  # eval_set=(F_test, y_test),
                  # early_stopping_rounds=20,
                  # verbose=0,
                  categorical_feature=cat_cols,
                  # eval_metric=eval_metric,
                  )
        if self.task == 'regression':
            y_pred = model.predict(F_test)
        else:
            y_pred = model.predict_proba(F_test)[:, 1]

        score = eval_metric(y_test, y_pred)
        return score

    def fit(self, X, y):
        columns = X.columns.to_list()
        if len(columns) > self.max_cols:
            columns = np.random.choice(columns, self.max_cols, replace=False)

        X_train, X_test, y_train, y_test = subsample(X, y,
                                                     max_samples=self.max_test_samples + self.max_train_samples,
                                                     train_samples=self.max_train_samples,
                                                     task=self.task)
        if self.task != 'regression' and y_train.dtype != 'int':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        cat_cols = column_object_category_bool(X_train)

        if len(cat_cols) > 0:
            X_train['__datacanvas__source__'] = 'train'
            X_test['__datacanvas__source__'] = 'test'
            X_all = pd.concat([X_train, X_test], axis=0)
            oe = OrdinalEncoder()
            X_all[cat_cols] = oe.fit_transform(X_all[cat_cols]).astype('int')

            X_train = X_all[X_all['__datacanvas__source__'] == 'train']
            X_test = X_all[X_all['__datacanvas__source__'] == 'test']
            X_train.pop('__datacanvas__source__')
            X_test.pop('__datacanvas__source__')

        self.scores_ = {}

        for c in columns:
            F_train = X_train[[c]]
            F_test = X_test[[c]]
            self.scores_[c] = self.feature_score(F_train, y_train, F_test, y_test)

        topn = np.min([np.max([int(len(columns) * self.ratio_max_cols), 10]), self.n_max_cols])

        sorted_scores = sorted([[col, score] for col, score in self.scores_.items()], key=lambda x: x[1])
        self.columns_ = [s[0] for s in sorted_scores[:topn]]

    def transform(self, X):
        return X[self.columns_]
