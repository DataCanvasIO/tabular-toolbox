# -*- coding:utf-8 -*-
import os

basedir = os.path.dirname(__file__)


def load_bank():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/bank-uci.csv.gz')
    return data


def load_bank_by_dask():
    from dask import dataframe as dd
    data = dd.read_csv(f'{basedir}/bank-uci.csv.gz', compression='gzip')
    return data
