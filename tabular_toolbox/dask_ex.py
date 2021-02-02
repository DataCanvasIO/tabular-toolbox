# -*- coding:utf-8 -*-
"""

"""
import math
import re
from collections import defaultdict
from functools import partial

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import model_selection as dm_sel, preprocessing as dm_pre, decomposition as dm_dec
from sklearn import inspection as sk_inspect, metrics as sk_metrics
from sklearn import model_selection as sk_sel, preprocessing as sk_pre, utils as sk_utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target

from .utils import logging

logger = logging.get_logger(__name__)

compute = dask.compute


def default_client():
    try:
        from dask.distributed import default_client as dask_default_client
        client = dask_default_client()
    except ValueError:
        client = None
    return client


def dask_enabled():
    return default_client() is not None


def is_local_dask():
    client = default_client()
    return type(client.cluster).__name__.lower().find('local') >= 0 if client is not None else False


def dask_worker_count():
    client = default_client()
    return len(client.ncores()) if client else 0


def is_dask_dataframe(X):
    return isinstance(X, dd.DataFrame)


def is_dask_series(X):
    return isinstance(X, dd.Series)


def is_dask_dataframe_or_series(X):
    return isinstance(X, (dd.DataFrame, dd.Series))


def is_dask_array(X):
    return isinstance(X, da.Array)


def is_dask_object(X):
    return isinstance(X, (da.Array, dd.DataFrame, dd.Series))


def exist_dask_object(*args):
    for a in args:
        if isinstance(a, (da.Array, dd.DataFrame, dd.Series)):
            return True
    return False


def exist_dask_dataframe(*args):
    for a in args:
        if isinstance(a, dd.DataFrame):
            return True
    return False


def exist_dask_array(*args):
    for a in args:
        if isinstance(a, da.Array):
            return True
    return False


def to_dask_type(X):
    if isinstance(X, np.ndarray):
        worker_count = dask_worker_count()
        chunk_size = math.ceil(X.shape[0] / worker_count) if worker_count > 0 else X.shape[0]
        X = da.from_array(X, chunks=chunk_size)
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        worker_count = dask_worker_count()
        partition_count = worker_count if worker_count > 0 else 1
        X = dd.from_pandas(X, npartitions=partition_count)

    return X


def _reset_part_index(df, start):
    new_index = pd.RangeIndex.from_range(range(start, start + df.shape[0]))
    df.index = new_index
    return df


def reset_index(X):
    assert isinstance(X, (pd.DataFrame, dd.DataFrame))

    if is_dask_dataframe(X):
        part_rows = X.map_partitions(lambda df: pd.DataFrame({'rows': [df.shape[0]]}),
                                     meta={'rows': 'int64'},
                                     ).compute()['rows'].tolist()
        assert len(part_rows) == X.npartitions

        divisions = [0]
        n = 0
        for i in part_rows:
            n += i
            divisions.append(n)
        divisions[-1] = divisions[-1] - 1

        delayed_reset_part_index = dask.delayed(_reset_part_index)
        parts = [delayed_reset_part_index(part, start) for part, start in zip(X.to_delayed(), divisions[0:-1])]
        X_new = dd.from_delayed(parts, divisions=divisions, meta=X.dtypes.to_dict())
        return X_new
    else:
        return X.reset_index(drop=True)


def make_chunk_size_known(a):
    assert is_dask_array(a)

    chunks = a.chunks
    if any(np.nan in d for d in chunks):
        if logger.is_debug_enabled():
            logger.debug(f'call extracted array compute_chunk_sizes, shape: {a.shape}')
        a = a.compute_chunk_sizes()
    return a


def make_divisions_known(X):
    assert is_dask_object(X)

    if is_dask_dataframe(X):
        if not X.known_divisions:
            columns = X.columns.tolist()
            X = X.reset_index()
            new_columns = X.columns.tolist()
            index_name = set(new_columns) - set(columns)
            X = X.set_index(list(index_name)[0] if index_name else 'index')
            assert X.known_divisions
    elif is_dask_series(X):
        if not X.known_divisions:
            X = make_divisions_known(X.to_frame())[X.name]
    else:  # dask array
        X = make_chunk_size_known(X)

    return X


def hstack_array(arrs):
    if all([a.ndim == 1 for a in arrs]):
        rows = compute(arrs[0].shape)[0][0]
        arrs = [a.reshape(rows, 1) if a.ndim == 1 else a for a in arrs]
    return stack_array(arrs, axis=1)


def vstack_array(arrs):
    return stack_array(arrs, axis=0)


def stack_array(arrs, axis=0):
    assert axis in (0, 1)
    ndims = set([len(a.shape) for a in arrs])
    if len(ndims) > 1:
        assert ndims == {1, 2}
        assert all([len(a.shape) == 1 or a.shape[1] == 1 for a in arrs])
        arrs = [a.reshape(compute(a.shape[0])[0], 1) if len(a.shape) == 1 else a for a in arrs]
    axis = min(axis, min([len(a.shape) for a in arrs]) - 1)
    assert axis >= 0

    if exist_dask_object(*arrs):
        arrs = [a.values if is_dask_dataframe_or_series(a) else a for a in map(to_dask_type, arrs)]
        if len(arrs) > 1:
            arrs = [make_chunk_size_known(a) for a in arrs]
        return da.concatenate(arrs, axis=axis)
    else:
        return np.concatenate(arrs, axis=axis)


def array_to_df(arrs, columns=None, meta=None):
    meta_df = None
    if isinstance(meta, (dd.DataFrame, pd.DataFrame)):
        meta_df = meta
        if columns is None:
            columns = meta_df.columns
        meta = dd.utils.make_meta(meta_df.dtypes.to_dict())
    elif isinstance(meta, (dd.Series, pd.Series)):
        meta_df = meta
        if columns is None:
            columns = meta_df.name
        meta = None

    df = dd.from_dask_array(arrs, columns=columns, meta=meta)

    if isinstance(meta_df, (dd.DataFrame, pd.DataFrame)):
        dtypes_src = meta_df.dtypes
        dtypes_dst = df.dtypes
        for col in meta_df.columns:
            if dtypes_src[col] != dtypes_dst[col]:
                df[col] = df[col].astype(dtypes_src[col])

    return df


def concat_df(dfs, axis=0, repartition=False, **kwargs):
    logger.info(f'[concat_df] enter with axis={axis}')
    if exist_dask_object(*dfs):
        dfs = [dd.from_dask_array(v) if is_dask_array(v) else v for v in dfs]
        if axis == 0:
            values = [df[dfs[0].columns].to_dask_array(lengths=True) for df in dfs]
            df = array_to_df(vstack_array(values), meta=dfs[0])
        else:
            dfs = [make_divisions_known(df) for df in dfs]
            df = dd.concat(dfs, axis=axis, **kwargs)

        if is_dask_series(dfs[0]) and df.name is None and dfs[0].name is not None:
            df.name = dfs[0].name
        if repartition:
            df = df.repartition(npartitions=dfs[0].npartitions)
    else:
        df = pd.concat(dfs, axis=axis, **kwargs)

    logger.info(f'[concat_df] done')
    return df


def train_test_split(*data, shuffle=True, random_state=None, **kwargs):
    if exist_dask_dataframe(*data):
        if len(data) > 1:
            data = [make_divisions_known(to_dask_type(x)) for x in data]
            head = data[0]
            for i in range(1, len(data)):
                if data[i].divisions != head.divisions:
                    print('-' * 10, f'repartition {i} from {data[i].divisions} to {head.divisions}')
                    data[i] = data[i].repartition(divisions=head.divisions)

        result = dm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)
    else:
        result = sk_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)

    return result


def fix_binary_predict_proba_result(proba):
    if is_dask_object(proba):
        if proba.ndim == 1:
            proba = make_chunk_size_known(proba)
            proba = proba.reshape((proba.size, 1))
        if proba.shape[1] == 1:
            proba = hstack_array([1 - proba, proba])
    else:
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T

    return proba


def wrap_for_local_scorer(estimator, target_type):
    def call_and_compute(fn_call, fn_fix, *args, **kwargs):
        r = fn_call(*args, **kwargs)
        if is_dask_object(r):
            r = r.compute()
            if callable(fn_fix):
                r = fn_fix(r)
        return r

    if hasattr(estimator, 'predict_proba'):
        orig_predict_proba = estimator.predict_proba
        fix = fix_binary_predict_proba_result if target_type == 'binary' else None
        setattr(estimator, '_orig_predict_proba', orig_predict_proba)
        setattr(estimator, 'predict_proba', partial(call_and_compute, orig_predict_proba, fix))

    if hasattr(estimator, 'predict'):
        orig_predict = estimator.predict
        setattr(estimator, '_orig_predict', orig_predict)
        setattr(estimator, 'predict', partial(call_and_compute, orig_predict, None))

    return estimator


def compute_and_call(fn_call, *args, **kwargs):
    if logger.is_debug_enabled():
        logger.debug(f'[compute_and_call] compute {len(args)} object')

    args = compute(*args, traverse=False)

    if logger.is_debug_enabled():
        logger.debug(f'[compute_and_call] call {fn_call.__name__}')
    # kwargs = {k: compute(v) if is_dask_array(v) else v for k, v in kwargs.items()}
    r = fn_call(*args, **kwargs)

    if logger.is_debug_enabled():
        logger.debug('[compute_and_call] to dask type')
    r = to_dask_type(r)

    if logger.is_debug_enabled():
        logger.debug('[compute_and_call] done')
    return r


def wrap_local_estimator(estimator):
    for fn_name in ('fit', 'predict', 'predict_proba'):
        fn_name_original = f'_wrapped_{fn_name}_by_wle'
        if hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original):
            fn = getattr(estimator, fn_name)
            assert callable(fn)
            setattr(estimator, fn_name_original, fn)
            setattr(estimator, fn_name, partial(compute_and_call, fn))

    return estimator


def permutation_importance(estimator, X, y, *args, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=None):
    if not is_dask_dataframe(X):
        return sk_inspect.permutation_importance(estimator, X, y, *args,
                                                 scoring=scoring,
                                                 n_repeats=n_repeats,
                                                 n_jobs=n_jobs,
                                                 random_state=random_state)
    random_state = sk_utils.check_random_state(random_state)

    def shuffle_partition(df, col_idx):
        shuffling_idx = np.arange(df.shape[0])
        random_state.shuffle(shuffling_idx)
        col = df.iloc[shuffling_idx, col_idx]
        col.index = df.index
        df.iloc[:, col_idx] = col
        return df

    if is_dask_object(y):
        y = y.compute()

    scorer = sk_metrics.check_scoring(wrap_for_local_scorer(estimator, type_of_target(y)), scoring)
    baseline_score = scorer(estimator, X, y)
    scores = []

    for c in range(X.shape[1]):
        col_scores = []
        for i in range(n_repeats):
            X_permuted = X.copy().map_partitions(shuffle_partition, c)
            col_scores.append(scorer(estimator, X_permuted, y))
        if logger.is_debug_enabled():
            logger.debug(f'permuted scores [{X.columns[c]}]: {col_scores}')
        scores.append(col_scores)

    importances = baseline_score - np.array(scores)
    return sk_utils.Bunch(importances_mean=np.mean(importances, axis=1),
                          importances_std=np.std(importances, axis=1),
                          importances=importances)


@sk_utils._deprecate_positional_args
def compute_class_weight(class_weight, *, classes, y):
    # f"""{sk_utils.class_weight.compute_class_weight.__doc__}"""
    if not is_dask_object(y):
        return sk_utils.class_weight.compute_class_weight(class_weight, classes, y)

    y = make_chunk_size_known(y)
    if set(compute(da.unique(y))[0]) - set(classes):
        raise ValueError("classes should include all valid labels that can be in y")

    if class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = dm_pre.LabelEncoder()
        y_ind = le.fit_transform(y)
        # if not all(np.in1d(classes, le.classes_)):
        #     raise ValueError("classes should have valid labels that are in y")
        # recip_freq = len(y) / (len(le.classes_) *
        #                        np.bincount(y_ind).astype(np.float64))
        # weight = recip_freq[le.transform(classes)]
        y_shape, y_ind_bincount, le_classes_ = compute(y.shape, da.bincount(y_ind), le.classes_)
        if not all(np.in1d(classes, le_classes_)):
            raise ValueError("classes should have valid labels that are in y")
        recip_freq = y_shape[0] / (len(le_classes_) * y_ind_bincount.astype(np.float64))
        weight = recip_freq[np.searchsorted(le_classes_, classes)]
    else:
        raise ValueError("Only class_weight == 'balanced' is supported.")

    return weight


def _compute_chunk_sample_weight(y, classes, classes_weights):
    t = np.ones(y.shape[0])
    for i, c in enumerate(classes):
        t[y == c] *= classes_weights[i]
    return t


def compute_sample_weight(y):
    assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)

    if is_dask_dataframe_or_series(y):
        y = y.values

    unique = compute(da.unique(y))[0] if is_dask_object(y) else np.unique(y)
    cw = list(compute_class_weight('balanced', unique, y))

    if is_dask_object(y):
        sample_weight = y.map_blocks(_compute_chunk_sample_weight, unique, cw, dtype=np.float64)
    else:
        sample_weight = _compute_chunk_sample_weight(y, unique, cw)

    return sample_weight


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2

        if isinstance(X, (pd.DataFrame, dd.DataFrame)):
            return self._fit_df(X, y)
        elif isinstance(X, (np.ndarray, da.Array)):
            return self._fit_array(X, y)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _fit_df(self, X, y=None):
        return self._fit_array(X.values, y.values if y else None)

    def _fit_array(self, X, y=None):
        n_features = X.shape[1]
        for n in range(n_features):
            le = dm_pre.LabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2

        if isinstance(X, (dd.DataFrame, pd.DataFrame)):
            return self._transform_dask_df(X)
        elif isinstance(X, (da.Array, np.ndarray)):
            return self._transform_dask_array(X)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _transform_dask_df(self, X):
        data = self._transform_dask_array(X.values)

        if isinstance(X, dd.DataFrame):
            result = dd.from_dask_array(data, columns=X.columns)
        else:
            result = pd.DataFrame(data, columns=X.columns)
        return result

    def _transform_dask_array(self, X):
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())

        data = []
        for n in range(n_features):
            data.append(self.encoders[n].transform(X[:, n]))

        if isinstance(X, da.Array):
            result = da.stack(data, axis=-1, allow_unknown_chunksizes=True)
        else:
            result = np.stack(data, axis=-1)

        return result

    # def fit_transform(self, X, y=None):
    #     return self.fit(X, y).transform(X)


class SafeOneHotEncoder(dm_pre.OneHotEncoder):
    def fit(self, X, y=None):
        if isinstance(X, (dd.DataFrame, pd.DataFrame)) and self.categories == "auto" \
                and any(d.name in {'object', 'bool'} for d in X.dtypes):
            a = []
            if isinstance(X, dd.DataFrame):
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype.name in {'object', 'bool'}:
                        Xi = Xi.astype('category').cat.as_known()
                    a.append(Xi)
                X = dd.concat(a, axis=1, ignore_unknown_divisions=True)
            else:
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype.name in {'object', 'bool'}:
                        Xi = Xi.astype('category')
                    a.append(Xi)
                X = pd.concat(a, axis=1)

        return super(SafeOneHotEncoder, self).fit(X, y)

    def get_feature_names(self, input_features=None):
        """
        Override this method to remove non-alphanumeric chars
        """
        # if not hasattr(self, 'drop_idx_'):
        #     setattr(self, 'drop_idx_', None)
        # return super(SafeOneHotEncoder, self).get_feature_names(input_features)

        # check_is_fitted(self)
        cats = self.categories_
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + '_' + str(idx) + '_' + re.sub('[^A-Za-z0-9_]+', '_', str(t))
                     for idx, t in enumerate(cats[i])]
            # if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
            #     names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)


class TruncatedSVD(dm_dec.TruncatedSVD):
    def fit_transform(self, X, y=None):
        X_orignal = X
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=2)

        if isinstance(X, dd.DataFrame):
            # y = y.values.compute_chunk_sizes() if y is not None else None
            r = super(TruncatedSVD, self).fit_transform(X.values.compute_chunk_sizes(), None)
        else:
            r = super(TruncatedSVD, self).fit_transform(X, y)

        if isinstance(X_orignal, (pd.DataFrame, np.ndarray)):
            r = r.compute()
        return r  # fixme, restore to DataFrame ??

    def transform(self, X, y=None):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).transform(X.values, y)

        return super(TruncatedSVD, self).transform(X, y)

    def inverse_transform(self, X):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).inverse_transform(X.values)

        return super(TruncatedSVD, self).inverse_transform(X)


class MaxAbsScaler(sk_pre.MaxAbsScaler):
    __doc__ = sk_pre.MaxAbsScaler.__doc__

    def fit(self, X, y=None, ):
        from dask_ml.utils import handle_zeros_in_scale

        self._reset()
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().fit(X, y)

        max_abs = X.reduction(lambda x: x.abs().max(),
                              aggregate=lambda x: x.max(),
                              token=self.__class__.__name__
                              ).compute()
        scale = handle_zeros_in_scale(max_abs)

        setattr(self, 'max_abs_', max_abs)
        setattr(self, 'scale_', scale)
        setattr(self, 'n_samples_seen_', 0)

        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(self, X, y=None, ):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None, ):
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)

        # Workaround for https://github.com/dask/dask/issues/2840
        if isinstance(X, dd.DataFrame):
            X = X.div(self.scale_)
        else:
            X = X / self.scale_
        return X

    def inverse_transform(self, X, y=None, copy=None, ):
        if not hasattr(self, "scale_"):
            raise Exception(
                "This %(name)s instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before "
                "using this method."
            )

        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().inverse_transform(X)

        if copy:
            X = X.copy()
        if isinstance(X, dd.DataFrame):
            X = X.mul(self.scale_)
        else:
            X = X * self.scale_

        return X


class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    __doc__ = r'Adapted from dask_ml OrdinalEncoder\n' + dm_pre.OrdinalEncoder.__doc__

    def __init__(self, columns=None, dtype=np.float64):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        self.dtypes_ = {c: X[c].dtype for c in X.columns}

        if self.columns is None:
            columns = X.select_dtypes(include=["category", 'object', 'bool']).columns
        else:
            columns = self.columns

        X = X.categorize(columns=columns)

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)
        self.categories_ = {c: X[c].cat.categories.sort_values() for c in columns}

        return self

    def transform(self, X, y=None):
        """Ordinal encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns)
            )

        encoder = self.make_encoder(self.categorical_columns_, self.categories_, self.dtype)
        if isinstance(X, pd.DataFrame):
            X = encoder(X)
        elif isinstance(X, dd.DataFrame):
            X = X.map_partitions(encoder)
        else:
            raise TypeError("Unexpected type {}".format(type(X)))

        return X

    def inverse_transform(self, X, missing_value=None):
        """Inverse ordinal-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        missing_value : skip doc

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        elif isinstance(X, da.Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = da.blockwise(len, "i", X[:, 0], "i", dtype="i8").compute()
                X = X.copy()
                chunks = (tuple(lengths), X.chunks[1])
                X._chunks = chunks
            X = dd.from_dask_array(X, columns=self.columns_)

        decoder = self.make_decoder(self.categorical_columns_, self.categories_, self.dtypes_)

        if isinstance(X, dd.DataFrame):
            X = X.map_partitions(decoder)
        else:
            X = decoder(X)

        return X

    @staticmethod
    def make_encoder(columns, categories, dtype):
        mappings = {}
        for col in columns:
            cat = categories[col]
            unseen = len(cat)
            m = defaultdict(dtype)
            for k, v in zip(cat, range(unseen)):
                m[k] = dtype(v + 1)
            mappings[col] = m

        def encode_column(x, c):
            return mappings[c][x]

        def safe_ordinal_encoder(pdf):
            assert isinstance(pdf, pd.DataFrame)

            pdf = pdf.copy()
            vf = np.vectorize(encode_column, excluded='c', otypes=[dtype])
            for col in columns:
                r = vf(pdf[col].values, col)
                if r.dtype != dtype:
                    # print(r.dtype, 'astype', dtype)
                    r = r.astype(dtype)
                pdf[col] = r
            return pdf

        return safe_ordinal_encoder

    @staticmethod
    def make_decoder(columns, categories, dtypes):
        def decode_column(x, col):
            cat = categories[col]
            xi = int(x)
            unseen = cat.shape[0]  # len(cat)
            if unseen >= xi >= 1:
                return cat[xi - 1]
            else:
                dtype = dtypes[col]
                if dtype in (np.float32, np.float64, np.float):
                    return np.nan
                elif dtype in (np.int32, np.int64, np.int, np.uint32, np.uint64, np.uint):
                    return -1
                else:
                    return None

        def safe_ordinal_decoder(pdf):
            assert isinstance(pdf, pd.DataFrame)

            pdf = pdf.copy()
            for col in columns:
                vf = np.vectorize(decode_column, excluded='col', otypes=[dtypes[col]])
                pdf[col] = vf(pdf[col].values, col)
            return pdf

        return safe_ordinal_decoder


class DataInterceptEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, fit=False, fit_transform=False, transform=False, inverse_transform=False):
        self._intercept_fit = fit
        self._intercept_fit_transform = fit_transform
        self._intercept_transform = transform
        self._intercept_inverse_transform = inverse_transform

        super(DataInterceptEncoder, self).__init__()

    def fit(self, X, *args, **kwargs):
        if self._intercept_fit:
            self.intercept(X, *args, **kwargs)

        return self

    def fit_transform(self, X, *args, **kwargs):
        if self._intercept_fit_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def transform(self, X, *args, **kwargs):
        if self._intercept_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def inverse_transform(self, X, *args, **kwargs):
        if self._intercept_inverse_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def intercept(self, X, *args, **kwargs):
        raise NotImplementedError()


class CallableAdapterEncoder(DataInterceptEncoder):
    def __init__(self, fn, **kwargs):
        assert callable(fn)

        self.fn = fn

        super(CallableAdapterEncoder, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        return self.fn(X, *args, **kwargs)


class DataCacher(DataInterceptEncoder):
    """
    persist and cache dask dataframe and array
    """

    def __init__(self, cache_dict, cache_key, remove_keys=None, **kwargs):
        assert isinstance(cache_dict, dict)

        if isinstance(remove_keys, str):
            remove_keys = set(remove_keys.split(','))

        self._cache_dict = cache_dict
        self.cache_key = cache_key
        self.remove_keys = remove_keys

        super(DataCacher, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        if self.cache_key:
            if isinstance(X, (dd.DataFrame, da.Array)):
                if logger.is_debug_enabled():
                    logger.debug(f'persist and cache {X._name} as {self.cache_key}')

                X = X.persist()

            self._cache_dict[self.cache_key] = X

        if self.remove_keys:
            for key in self.remove_keys:
                if key in self._cache_dict.keys():
                    if logger.is_debug_enabled():
                        logger.debug(f'remove cache {key}')
                    del self._cache_dict[key]

        return X

    @property
    def cache_dict(self):
        return list(self._cache_dict.keys())


class CacheCleaner(DataInterceptEncoder):

    def __init__(self, cache_dict, **kwargs):
        assert isinstance(cache_dict, dict)

        self._cache_dict = cache_dict

        super(CacheCleaner, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        if logger.is_debug_enabled():
            logger.debug(f'clean cache with {list(self._cache_dict.keys())}')
        self._cache_dict.clear()

        return X

    @property
    def cache_dict(self):
        return list(self._cache_dict.keys())

    # # override this to remove 'cache_dict' from estimator __expr__
    # @classmethod
    # def _get_param_names(cls):
    #     params = super()._get_param_names()
    #     return [p for p in params if p != 'cache_dict']
