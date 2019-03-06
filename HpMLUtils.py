"""Module with several helper functions for H+ machine learning algorithms"""

import numpy as np
from itertools import chain
from sklearn.utils import safe_indexing, check_random_state,check_array
from sklearn.utils.validation import check_is_fitted, column_or_1d, FLOAT_DTYPES
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator, TransformerMixin

def getXy():
    """return an example input (features and classes) for test-train validation """
    
    y=np.linspace(0.1,1,10)
    X=np.zeros((10,10))
    for i in range(0,10):
        for j in range(0,10):
            X[i,j]=i*10+j
    print X,y
    return X,y

def summarizeFitData(X, y, w=None, categories=None, showavevarminmax=True):
    """ prints a summary of the X=features, y=classes, w=weights data on the command line"""
    
    print "X.shape=", X.shape, "y.shape=", y.shape,
    if w is None:
        w=pd.Series(np.ones(y.shape))
    else:
        print "w.shape=", w.shape,

    print
    print "columns=", X.columns
    
    if categories is None:
        categories=y

    uniquecategories=sorted(categories.unique())
    print "categories=",uniquecategories
    print 
    
    print "sum of weights per category"
    length=max([len(str(x)) for x in uniquecategories]+[10])
    print ('{:>'+str(length)+'}').format("all"),('{:>'+str(length)+'}').format(w.sum())
    for cat in uniquecategories:
        print ('{:>'+str(length)+'}').format(cat), ('{:>'+str(length)+'}').format(w[categories==cat].sum())
    print "\n"

    if showavevarminmax:
        print "average"
        variablelength=max([len(x) for x in X.columns]+[len("variable/class")])
        print ('{:>'+str(variablelength)+'}').format("variable/class"),
        print ('{:>'+str(length)+'}').format("all"),
        for cat in uniquecategories:
            print ('{:>'+str(length)+'}').format(cat),
        print
    
        for i,variable in enumerate(X.columns):
            print ('{:>'+str(variablelength)+'}').format(variable),
            print ('{:>'+str(length)+'.3}').format(np.average(X[variable], weights=w)),
            for cat in uniquecategories:
                print ('{:>'+str(length)+'.3}').format(np.average(X[variable][categories==cat], weights=w[categories==cat])),
            print
        print "\n"
        
        print "variance"
        print ('{:>'+str(variablelength)+'}').format("variable/class"),
        print ('{:>'+str(length)+'}').format("all"),
        for cat in uniquecategories:
            print ('{:>'+str(length)+'}').format(cat),
        print
    
        for i,variable in enumerate(X.columns):
            print ('{:>'+str(variablelength)+'}').format(variable),
            print ('{:>'+str(length)+'.3}').format(variance(X[variable], weights=w)),
            for cat in uniquecategories:
                print ('{:>'+str(length)+'.3}').format(variance(X[variable][categories==cat], weights=w[categories==cat])),
            print
        print "\n"

        print "min/max"
        print ('{:>'+str(variablelength)+'}').format("variable/class"),
        print ('{:>'+str(length)+'}').format("all/min"),
        print ('{:>'+str(length)+'}').format("all/max"),
        for cat in uniquecategories:
            print ('{:>'+str(length)+'}').format(str(cat)+"/min"),
            print ('{:>'+str(length)+'}').format(str(cat)+"/max"),
        print
    
        for i,variable in enumerate(X.columns):
            print ('{:>'+str(variablelength)+'}').format(variable),
            print ('{:>'+str(length)+'.3}').format(float(np.min(X[variable]))),
            print ('{:>'+str(length)+'.3}').format(float(np.max(X[variable]))),
            for cat in uniquecategories:
                print ('{:>'+str(length)+'.3}').format(float(np.min(X[variable][categories==cat]))),
                print ('{:>'+str(length)+'.3}').format(float(np.max(X[variable][categories==cat]))),
            print
        print "\n"
    
def variance(values, weights=None, axis=0):
    """ returns weighted (biased) variance
        values: array/series with values
        weights: array/series with weights (same dimension as values)
    """
    
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return variance

def sqrtvariance(values, weights=None, axis=0):
    """ returns sqare root of weighted (biased) variance
        values: array/series with values
        weights: array/series with weights (same dimension as values)
    """
    
    return np.sqrt(variance(values, weights=weights, axis=axis))

class PredefinedThreeSplit(BaseCrossValidator):
    """Predefined split cross-validator into three datasets: test(value=0), train(value=1), eval(value=2) (eval indices are the indices which are not returned)"""

    def __init__(self, test_fold, shuffle=True, random_state=None):
        """ constructor
        test_fold: series with values 0,1,2 to which dataset (testing=0, training=1, evalution=2) an event belongs
        """
        
        self.test_fold = np.array(test_fold, dtype=np.int)
        self.test_fold = column_or_1d(self.test_fold)
        self.shuffle=shuffle
        if shuffle:
            self.rng = check_random_state(random_state)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        for train_index in [0,1]:
            train_indices=np.where(self.test_fold==train_index)[0]
            test_indices=np.where(self.test_fold==(train_index+1)%2)[0]
            if self.shuffle:
                self.rng.shuffle(train_indices)
                self.rng.shuffle(test_indices)
            yield train_indices, test_indices        

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator, here always 2
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator, here always 2.
        """

        return 2

def sample(*arrays, **options):
    """sample several arrays at the same time

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    frac : float
        fraction of rows to sample
    categorytosample: object
        only sample rows with this category
    categories: sequence with same length as arrays
        categories to compare with categorytosample

    Returns
    -------
    splitting : list, length=len(arrays)
        List containing sampled inputs.
    """
    
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    random_state = options.pop('random_state', None)

    frac = options.pop('frac', 0.5)

    categorytosample = options.pop('categorytosample', None)
    categories = options.pop('categories', None)
    if not categorytosample is None and categories is None:
        raise ValueError("Categories have to be provided if sampling by category is requested.")

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    rng = check_random_state(random_state)
    if categorytosample is None:
        maxidx=arrays[0].shape[0]
        nrows=int(maxidx*frac)
        indicestosample=np.linspace(0,maxidx-1,num=maxidx)
        indicesnottosample=np.array([])
    else:
        cat=categories.copy().reset_index(drop=True)
        indicestosample=np.array(cat[cat==categorytosample].index)
        indicesnottosample=np.array(cat[cat!=categorytosample].index)
        nrows=int(indicestosample.shape[0]*frac)
        
    rng.shuffle(indicestosample)
    indices=np.sort(np.concatenate((indicestosample[:nrows],indicesnottosample)))

    return list(safe_indexing(a, indices) for a in arrays)
    
def train_test_split3(*arrays, **options):
    """Split arrays or matrices into random train, test and eval subsets
    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test-eval split of inputs.
    """

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    random_state = options.pop('random_state', None)
    shuffleresults = options.pop('shuffle', True)    
    test_fold  = options.pop('test_fold', None)
    if test_fold is None:
        raise TypeError("Parameter test_fold is required.")

    test_fold = np.array(test_fold, dtype=np.int)
    test_fold = column_or_1d(test_fold)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    evalu=np.where(test_fold==2)[0]
    if shuffleresults:
        rng = check_random_state(random_state)
        rng.shuffle(evalu)
    cv = PredefinedThreeSplit(test_fold=test_fold, shuffle=shuffleresults, random_state=random_state)
    train, test = next(cv.split())

    #print evalu
    if len(evalu)==0:
        return list(chain.from_iterable((safe_indexing(a, train),
                                         safe_indexing(a, test), np.array(0)) for a in arrays))
    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test),
                                     safe_indexing(a, evalu)) for a in arrays))

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
         
    def transform(self, X, y='deprecated', copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        y : (ignored)
        """   

        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        #X = check_array(X, copy=copy, warn_on_dtype=True,
        #                estimator=self, dtype=FLOAT_DTYPES,
        #                force_all_finite='allow-nan')

        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """ inverse transformation (see transform)"""
        
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        #X = check_array(X, copy=copy, warn_on_dtype=True,
        #                estimator=self, dtype=FLOAT_DTYPES,
        #                force_all_finite='allow-nan')

        if self.with_mean:
            X += self.mean_
        if self.with_std:
            X *= self.scale_
        return X

class MultiSWeightsScaler():
    """ Class that scales makes the integral of the signal weights be 1., for several signal categories the distribution as a function of the class variable is flattened. Background is not considered."""

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_

    def fit(self,X,y, sample_weight):
        """learns the sum of weights for all classes and calculates a scale factor for each class so that the sum of weights for the signal is flattened as a function of the class variable (and the integral is 1.0)
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        classes=sorted(y.unique())
        differences={}
        #set the differences between signal points
        differences={classes[i]:(classes[i+1]-classes[i-1])/2 for i in range(1,len(classes)-1) if classes[i]>0}
        differences[classes[0]]=classes[1]-classes[0]
        differences[classes[-1]]=classes[-1]-classes[-2]
        diffsum=sum(differences.values())
        #print differences, "->", diffsum
        self.scale_={}
        for classlabel in classes:
            sumweight=sample_weight[y==classlabel].sum()
            self.scale_[classlabel]=differences[classlabel]/(sumweight*diffsum)
        return
        
    def transform(self, X, y, sample_weight, copy=None):
        """Transforms the sum of weights for all classes so that sum of weights for the signal is flattened as a function of the class variable (and the integral is 1.0)
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        
        for classlabel in self.scale_:
            sample_weight[y==classlabel]*=self.scale_[classlabel]
        return X

class MultiSBWeightsScaler():
    """ Class that scales makes the integral of the signal/background weights be 0.5, for several signal categories the distribution as a function of the class variable is flattened"""

    def __init__(self, backgroundclass=0):
        """ constructor
            backgroundclass: label for background
        """
        self.backgroundclass=backgroundclass
    
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_

    def fit(self,X,y, sample_weight):
        """learns the sum of weights for all classes and calculates a scale factor for each class so that sum of weight for background is 0.5 and the sum of weights for the signal is flattened as a function of the class variable (and the integral is 0.5)
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        classes=sorted(y.unique())
        classes.remove(self.backgroundclass)
        differences={}
        if len(classes)>1: #more than 1 signal
            #set the differences between signal points
            differences={classes[i]:(classes[i+1]-classes[i-1])/2 for i in range(1,len(classes)-1) if classes[i]>0}
            differences[classes[0]]=classes[1]-classes[0]
            differences[classes[-1]]=classes[-1]-classes[-2]
            diffsum=sum(differences.values())
            #print differences, "->", diffsum
        else:
            differences[classes[0]]=1
            diffsum=1
        self.scale_={}
        for classlabel in classes:
            sumweight=sample_weight[y==classlabel].sum()
            self.scale_[classlabel]=differences[classlabel]/(2*sumweight*diffsum)
        sumweight=sample_weight[y==self.backgroundclass].sum()
        self.scale_[self.backgroundclass]=0.5/sumweight
        return
        
    def transform(self, X, y, sample_weight, copy=None):
        """Transforms the sum of weights for all classes so that sum of weight for background is 0.5 and the sum of weights for the signal is flattened as a function of the class variable (and the integral is 0.5)
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        
        for classlabel in self.scale_:
            sample_weight[y==classlabel]*=self.scale_[classlabel]
        return X
        
        
