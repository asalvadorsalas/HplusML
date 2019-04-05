"""Module with classes for the scaling and normalisation of ML feature vector for H+ machine learning algorithms"""

from __future__ import print_function
from sklearn.base import BaseEstimator, TransformerMixin
from HpMLUtils import variance
from sklearn.utils.validation import check_is_fitted
import numpy as np


class WeightedStandardScaler(BaseEstimator, TransformerMixin):
    """Class which transforms all features to have average 0 and variance 1, same as scikit-learn StandardScaler, but taking weights into account """
        
    def __init__(self, copy=True, with_mean=True, with_std=True):
        """ with_mean: boolean, if true transfroms weighted average to 0
            with_std: boolean, if true transforms weighted variance to 1
            copy: boolean, if true copies data, if false change data in place
        """
        
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()

        if self.with_mean:
            self.mean_ = np.average(X,axis=0,weights=sample_weight)
        if self.with_std:
            self.var_ = variance(X,weights=sample_weight)
            #np.average((X-self.mean_)*(X-self.mean_),axis=0, weights=sample_weight)
            self.scale_ = np.sqrt(self.var_)

        return self
         
    def transform(self, X, y='deprecated', sample_weight=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        y : (not affected)
        sample_weight : (not affected)
        """   

        check_is_fitted(self, 'scale_')

        #copy = copy if copy is not None else self.copy

        #X = check_array(X, copy=copy, warn_on_dtype=True,
        #                estimator=self, dtype=FLOAT_DTYPES,
        #                force_all_finite='allow-nan')

        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X, y, sample_weight

    def inverse_transform(self, X, y='deprecated', sample_weight=None):
        """ inverse transformation (see transform)"""
        
        check_is_fitted(self, 'scale_')

        #copy = copy if copy is not None else self.copy

        #X = check_array(X, copy=copy, warn_on_dtype=True,
        #                estimator=self, dtype=FLOAT_DTYPES,
        #                force_all_finite='allow-nan')

        if self.with_mean:
            X += self.mean_
        if self.with_std:
            X *= self.scale_
        return X, y, sample_weight

class WeightedStandardScalerForHp(WeightedStandardScaler):
    """ Same as WeightedStandardScaler however having a special transformation for njets and nbjets (as those are integer variables we just divide by 10)"""

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()

        if self.with_mean:
            self.mean_ = np.average(X,axis=0,weights=sample_weight)
        if self.with_std:
            self.var_ = variance(X,weights=sample_weight)
            #np.average((X-self.mean_)*(X-self.mean_),axis=0, weights=sample_weight)
            self.scale_ = np.sqrt(self.var_)

        for i,col in enumerate(X.columns):
            if col=="nJets" or "nBTags" in col:
                self.mean_[i]=0
                self.var_[i]=100
                self.scale_[i]=10

        return self
