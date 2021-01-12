# -*- coding:utf-8 -*-
import os

basedir = os.path.dirname(__file__)


def load_bank():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/bank-uci.csv.gz')
    return data


def load_bank_by_dask():
    from dask import dataframe as dd
    data = dd.read_csv(f'{basedir}/bank-uci.csv.gz', compression='gzip', blocksize=None)
    return data


def load_glass_uci():
    import pandas as pd
    #print(f'Base dir:{basedir}')
    data = pd.read_csv(f'{basedir}/glass_uci.csv', header=None)
    return data

def load_blood():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/blood.csv')
    return data

def load_telescope():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/telescope.csv')
    return data

def load_Bike_Sharing():
    import pandas as pd
    data = pd.read_csv(f'{basedir}/Bike_Sharing.csv')
    return data
