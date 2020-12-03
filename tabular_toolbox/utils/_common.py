import hashlib

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..const import *


def infer_task_type(y):
    if len(y.shape) > 1 and y.shape[-1] > 1:
        labels = list(range(y.shape[-1]))
        task = 'multilable'
        return task, labels

    uniques = set(y)
    n_unique = len(uniques)
    labels = []

    if n_unique == 2:
        print(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
        task = TASK_BINARY  # TASK_BINARY
        labels = sorted(uniques)
    else:
        if y.dtype == 'float':
            print(f'Target column type is float, so inferred as a [regression] task.')
            task = TASK_REGRESSION
        else:
            if n_unique > 1000:
                if 'int' in y.dtype:
                    print(
                        'The number of classes exceeds 1000 and column type is int, so inferred as a [regression] task ')
                    task = TASK_REGRESSION
                else:
                    raise ValueError(
                        'The number of classes exceeds 1000, please confirm whether your predict target is correct ')
            else:
                print(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                task = TASK_MULTICLASS
                labels = sorted(uniques)
    return task, labels


def hash_dataframe(df, method='md5', index=False):
    assert isinstance(df, (pd.DataFrame, dd.DataFrame))

    m = getattr(hashlib, method)()

    for col in df.columns:
        m.update(col.encode())

    if isinstance(df, dd.DataFrame):
        x = df.map_partitions(lambda part: pd.util.hash_pandas_object(part, index=index),
                              meta=(None, 'u8'))
        percentiles = da.percentile(x.values, range(0, 100)).compute()
    else:
        x = pd.util.hash_pandas_object(df, index=index)
        percentiles = np.percentile(x.values, range(0, 100))

    for n in percentiles:
        # m.update(struct.pack('d', n))
        m.update(n)

    return m.hexdigest()
