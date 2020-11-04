# -*- coding:utf-8 -*-
"""

"""
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import decomposition as dm_dec
from dask_ml import preprocessing as dm_pre
from sklearn import preprocessing as sk_pre
from sklearn.base import BaseEstimator, TransformerMixin


def to_dask_type(X):
    if isinstance(X, np.ndarray):
        X = da.from_array(X)
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        X = dd.from_pandas(X, npartitions=2)

    return X


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
        if y is not None:
            return self._fit_array(X.values, y.values)
        else:
            return self._fit_array(X.values)

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


class OneHotEncoder(dm_pre.OneHotEncoder):
    def fit(self, X, y=None):
        if isinstance(X, (dd.DataFrame, pd.DataFrame)) and self.categories == "auto" \
                and any(d.name == 'object' for d in X.dtypes):
            a = []
            if isinstance(X, dd.DataFrame):
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype == 'object':
                        Xi = Xi.astype('category').cat.as_known()
                    a.append(Xi)
                X = dd.concat(a, axis=1)
            else:
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype == 'object':
                        Xi = Xi.astype('category')
                    a.append(Xi)
                X = pd.concat(a, axis=1)

        return super(OneHotEncoder, self).fit(X, y)

    def get_feature_names(self, input_features=None):
        if not hasattr(self, 'drop_idx_'):
            setattr(self, 'drop_idx_', None)
        return super(OneHotEncoder, self).get_feature_names(input_features)


class TruncatedSVD(dm_dec.TruncatedSVD):
    def fit_transform(self, X, y=None):
        X_orignal = X
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=2)

        if isinstance(X, dd.DataFrame):
            r = super(TruncatedSVD, self).fit_transform(X.values, y)
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
            return super().partial_fit(X, y)

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
            columns = X.select_dtypes(include=["category", 'object']).columns
        else:
            columns = self.columns

        X = X.categorize(columns=columns)

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)
        self.categories_ = {c: X[c].cat.categories.sort_values() for c in columns}

        def make_encoder(columns, categories, dtype):
            encoders = {}
            for col in columns:
                cat = categories[col]
                unseen = len(cat)
                m = dict(zip(cat, range(unseen)))
                vf = np.vectorize(lambda x: m[x] if x in m.keys() else unseen)
                encoders[col] = vf

            def pdf_encoder(pdf):
                assert isinstance(pdf, pd.DataFrame)

                pdf = pdf.copy()
                for col in columns:
                    r = encoders[col](pdf[col].values)
                    if r.dtype != dtype:
                        r = r.astype(dtype)
                    pdf[col] = r
                return pdf

            return pdf_encoder

        def make_decoder(columns, categories, dtypes):
            decoders = {}
            for col in columns:
                dtype = dtypes[col]
                if dtype in (np.float32, np.float64, np.float):
                    default_value = np.nan
                elif dtype in (np.int32, np.int64, np.int, np.uint32, np.uint64, np.uint):
                    default_value = -1
                else:
                    default_value = None
                    dtype = np.object

                cat = categories[col]
                unseen = cat.shape[0]  # len(cat)
                vf = np.vectorize(lambda x: cat[x] if unseen > x >= 0 else default_value,
                                  otypes=[dtype])
                decoders[col] = vf

            def pdf_decoder(pdf):
                assert isinstance(pdf, pd.DataFrame)

                pdf = pdf.copy()
                for col in columns:
                    pdf[col] = decoders[col](pdf[col].values)
                return pdf

            return pdf_decoder

        self.encoder_ = make_encoder(self.categorical_columns_, self.categories_, self.dtype)
        self.decoder_ = make_decoder(self.categorical_columns_, self.categories_, self.dtypes_)
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

        if isinstance(X, pd.DataFrame):
            X = self.encoder_(X)
        elif isinstance(X, dd.DataFrame):
            X = X.map_partitions(self.encoder_)
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

        if isinstance(X, dd.DataFrame):
            X = X.map_partitions(self.decoder_)
        else:
            X = self.decoder_(X)

        return X
