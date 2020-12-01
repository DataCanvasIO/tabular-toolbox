import hashlib

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd


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
