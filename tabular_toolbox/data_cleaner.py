# -*- coding:utf-8 -*-
"""

"""
import copy

import numpy as np
import pandas as pd
from dask import dataframe as dd

from .column_selector import column_object, column_int
from .utils import logging

logger = logging.get_logger(__name__)


def _reduce_mem_usage(df, verbose=True):
    """
    Adaption from :https://blog.csdn.net/xckkcxxck/article/details/88170281
    :param verbose:
    :return:
    """
    if isinstance(df, dd.DataFrame):
        raise Exception('"reduce_mem_usage" is not supported for Dask DataFrame.')

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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))


def _drop_duplicated_columns(X):
    if isinstance(X, dd.DataFrame):
        duplicates = X.reduction(chunk=lambda c: pd.DataFrame(c.T.duplicated()).T,
                                 aggregate=lambda a: np.all(a, axis=0)).compute()
    else:
        duplicates = X.T.duplicated()

    columns = [i for i, v in duplicates.items() if not v]
    X = X[columns]

    return X


def _correct_object_dtype(X):
    object_columns = column_object(X)

    if isinstance(X, dd.DataFrame):
        def detect_dtype(df):
            result = {}
            df = df.copy()
            for col in object_columns:
                try:
                    df[col] = df[col].astype('float')
                    result[col] = [True]  # float-able
                except:
                    result[col] = [False]
            return pd.DataFrame(result)

        floatable = X.reduction(chunk=detect_dtype,
                                aggregate=lambda a: np.all(a, axis=0)).compute()
        float_columns = [i for i, v in floatable.items() if v]
        for col in float_columns:
            X[col] = X[col].astype('float')
        logger.info(f'Correct columns [{",".join(float_columns)}] to float.')
    else:
        for col in object_columns:
            try:
                X[col] = X[col].astype('float')
            except Exception as e:
                logger.error(f'Correct object column [{col}] failed. {e}')

    return X


def _drop_constant_columns(X):
    if isinstance(X, dd.DataFrame):
        nunique = X.reduction(chunk=lambda c: pd.DataFrame(c.nunique(dropna=True)).T,
                              aggregate=np.max).compute()
    else:
        nunique = X.nunique(dropna=True)

    columns = [i for i, v in nunique.items() if v > 1]
    X = X[columns]

    return X


class DataCleaner:
    def __init__(self, nan_chars=None, correct_object_dtype=True, drop_constant_columns=True,
                 drop_duplicated_columns=False, drop_label_nan_rows=True, replace_inf_values=np.nan,
                 drop_columns=None, reduce_mem_usage=False,
                 int_convert_to='float'):
        self.nan_chars = nan_chars
        self.correct_object_dtype = correct_object_dtype
        self.drop_constant_columns = drop_constant_columns
        self.drop_label_nan_rows = drop_label_nan_rows
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.drop_duplicated_columns = drop_duplicated_columns
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to
        self.df_meta_ = None

    def clean_data(self, X, y):
        assert isinstance(X, (pd.DataFrame, dd.DataFrame))

        if y is not None:
            y_name = '__tabular-toolbox__Y__'
            # X.insert(0, y_name, y)
            X[y_name] = y

        if self.nan_chars is not None:
            logger.info(f'Replace chars{self.nan_chars} to NaN')
            X = X.replace(self.nan_chars, np.nan)

        if self.correct_object_dtype:
            logger.info('Correct data type for object columns.')
            # for col in column_object(X):
            #     try:
            #         X[col] = X[col].astype('float')
            #     except Exception as e:
            #         logger.error(f'Correct object column [{col}] failed. {e}')
            X = _correct_object_dtype(X)

        if self.drop_duplicated_columns:
            # duplicates = X.T.duplicated().values
            # columns = [c for i, c in enumerate(X.columns.to_list()) if not duplicates[i]]
            # X = X[columns]
            X = _drop_duplicated_columns(X)

        if self.int_convert_to is not None:
            logger.info(f'Convert int type to {self.int_convert_to}')
            int_cols = column_int(X)
            X[int_cols] = X[int_cols].astype(self.int_convert_to)

        if y is not None:
            if self.drop_label_nan_rows:
                logger.info('Clean the rows which label is NaN')
                X = X.dropna(subset=[y_name])
            y = X.pop(y_name)

        if self.drop_columns is not None:
            logger.info(f'Drop columns:{self.drop_columns}')
            for col in self.drop_columns:
                X.pop(col)

        if self.drop_constant_columns:
            logger.info('Clean invalidate columns')
            # for col in X.columns:
            #     n_unique = X[col].nunique(dropna=True)
            #     if n_unique <= 1:
            #         X.pop(col)
            X = _drop_constant_columns(X)

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
            _reduce_mem_usage(X)

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
        self.df_meta_ = df_meta
        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)
        X, y = self.clean_data(X, y)
        if self.df_meta_ is not None:
            logger.info('Processing with meta info')
            all_cols = []
            for dtype, cols in self.df_meta_.items():
                all_cols += cols
                X[cols] = X[cols].astype(dtype)
            drop_cols = set(X.columns.to_list()) - set(all_cols)
            X = X[all_cols]
            logger.info(f'droped columns:{drop_cols}')

        if self.replace_inf_values is not None:
            logger.info(f'Replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)
        if y is None:
            return X
        else:
            return X, y

    def append_drop_columns(self, columns):
        if self.df_meta_ is None:
            if self.drop_columns is None:
                self.drop_columns = []
            self.drop_columns = list(set(self.drop_columns + columns))
        else:
            meta = {}
            for dtype, cols in self.df_meta_.items():
                meta[dtype] = [c for c in cols if c not in columns]
            self.df_meta_ = meta
