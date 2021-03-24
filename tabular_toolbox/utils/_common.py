import hashlib
import pickle
from functools import partial
from io import BytesIO

import dask.dataframe as dd
import numpy as np
import pandas as pd

from . import logging
from ..const import *

logger = logging.get_logger(__name__)


def infer_task_type(y):
    if len(y.shape) > 1 and y.shape[-1] > 1:
        labels = list(range(y.shape[-1]))
        task = 'multilable'
        return task, labels

    if hasattr(y, 'unique'):
        uniques = set(y.unique())
    else:
        uniques = set(y)

    if uniques.__contains__(np.nan):
        uniques.remove(np.nan)
    n_unique = len(uniques)
    labels = []

    if n_unique == 2:
        logger.info(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
        task = TASK_BINARY  # TASK_BINARY
        labels = sorted(uniques)
    else:
        if y.dtype == 'float':
            logger.info(f'Target column type is float, so inferred as a [regression] task.')
            task = TASK_REGRESSION
        else:
            if n_unique > 1000:
                if 'int' in y.dtype:
                    logger.info(
                        'The number of classes exceeds 1000 and column type is int, so inferred as a [regression] task ')
                    task = TASK_REGRESSION
                else:
                    raise ValueError(
                        'The number of classes exceeds 1000, please confirm whether your predict target is correct ')
            else:
                logger.info(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                task = TASK_MULTICLASS
                labels = sorted(uniques)
    return task, labels


def hash_dataframe(df, method='md5', index=False):
    assert isinstance(df, (pd.DataFrame, dd.DataFrame))

    m = getattr(hashlib, method)()

    for col in df.columns:
        m.update(str(col).encode())

    if isinstance(df, dd.DataFrame):
        x = df.map_partitions(lambda part: pd.util.hash_pandas_object(part, index=index),
                              meta=(None, 'u8')).compute()
    else:
        x = pd.util.hash_pandas_object(df, index=index)

    np.vectorize(m.update, otypes=[None], signature='()->()')(x.values)

    return m.hexdigest()


def hash_data(data, method='md5'):
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        return hash_dataframe(data, method=method)
    elif isinstance(data, (pd.Series, dd.Series)):
        return hash_dataframe(data.to_frame(), method=method)

    if isinstance(data, (bytes, bytearray)):
        pass
    elif isinstance(data, str):
        data = data.encode('utf-8')
    else:
        if isinstance(data, (list, tuple)):
            data = [hash_data(x) if x is not None else x for x in data]
        elif isinstance(data, dict):
            data = {hash_data(k): hash_data(v) if v is not None else v for k, v in data.items()}
        buf = BytesIO()
        pickle.dump(data, buf)
        data = buf.getvalue()
        buf.close()

    m = getattr(hashlib, method)()
    m.update(data)
    return m.hexdigest()


def load_data(data, **kwargs):
    assert isinstance(data, (str, pd.DataFrame, dd.DataFrame))

    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        return data

    import os.path as path
    import glob

    try:
        from dask.distributed import default_client as dask_default_client
        client = dask_default_client()
        dask_enabled, worker_count = True, len(client.ncores())
    except ValueError:
        dask_enabled, worker_count = False, 1

    fmt_mapping = {
        'csv': (partial(pd.read_csv, low_memory=False), dd.read_csv),
        'txt': (pd.read_csv, dd.read_csv),
        'parquet': (pd.read_parquet, dd.read_parquet),
        'par': (pd.read_parquet, dd.read_parquet),
        'json': (pd.read_json, dd.read_json),
        'pkl': (pd.read_pickle, None),
        'pickle': (pd.read_pickle, None),
    }

    def get_file_format(file_path):
        return path.splitext(file_path)[-1].lstrip('.')

    def get_file_format_by_glob(data_pattern):
        for f in glob.glob(data_pattern, recursive=True):
            fmt_ = get_file_format(f)
            if fmt_ in fmt_mapping.keys():
                return fmt_
        return None

    if glob.has_magic(data):
        fmt = get_file_format_by_glob(data)
    elif not path.exists(data):
        raise ValueError(f'Not found path {data}')
    elif path.isdir(data):
        path_pattern = f'{data}*' if data.endswith(path.sep) else f'{data}{path.sep}*'
        fmt = get_file_format_by_glob(path_pattern)
    else:
        fmt = path.splitext(data)[-1].lstrip('.')

    if fmt not in fmt_mapping.keys():
        # fmt = fmt_mapping.keys()[0]
        raise ValueError(f'Not supported data format{fmt}')
    fn = fmt_mapping[fmt][int(dask_enabled)]
    if fn is None:
        raise ValueError(f'Not supported data format{fmt}')

    if dask_enabled and path.isdir(data) and not glob.has_magic(data):
        data = f'{data}*' if data.endswith(path.sep) else f'{data}{path.sep}*'
    df = fn(data, **kwargs)

    if dask_enabled and worker_count > 1 and df.npartitions < worker_count:
        df = df.repartition(npartitions=worker_count)

    return df
