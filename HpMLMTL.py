"""Module which implements function for H+ multi-task learning"""

from sklearn.utils import shuffle
from scipy.stats import rv_discrete
import pandas as pd
import numpy as np

class HpMTLBackgroundRandomizer():
    """ Class that assigns random signal mass hypotheses to background training data with the PDF learned from signal for H+ multi-task learning"""

    def __init__(self, backgroundclass=0, verbose=False):
        """ constructor
            backgroundclass: label for background
            verbose: print debug information if True
        """
        self.backgroundclass=backgroundclass
        self.verbose=verbose
        self.randomseed=123456789
        self.xk=[]
        self.pk=[]

    def fit(self, X, y, sample_weight):
        """ determines the mass PDF for signal events
            X: feature vector
            y: hp mass (background class for background)
            sample_weight: weights for events
        """
        signalsum=np.sum(sample_weight[y!=self.backgroundclass])
        for name, group in sample_weight.groupby(y):
            if name!=self.backgroundclass:
                self.xk.append((-1)*name)
                self.pk.append(group.sum()/signalsum)
        if self.verbose:
            print("Signal PDF:",self.xk,self.pk)

    def transform(self, X, y, sample_weight):
        """ randomly assigns signal labels according to fitted pdf
            X: feature vector
            y: hp mass (background class for background)
            sample_weight: weights for events
        """

        np.random.seed(seed=self.randomseed+len(y)) #add len(y) not not have the same random seed for test and train
        custm=rv_discrete(values=(self.xk,self.pk))
        labels=y.apply(lambda val: val if val!=self.backgroundclass else custm.rvs())
        X.hpmass=labels.abs()
        if self.verbose and not sample_weight is None:
            print("the following is the difference between + and - mass")
            print((sample_weight*((labels>0)-0.5)*2).groupby(labels.abs()).sum())
            print("the following is the sum of weights")
            print(sample_weight.groupby(labels).sum())

        return X, labels, sample_weight
        
    def fitAndTransform(self,X, y, sample_weight):
        """ fit and transform at the same time
            X: feature vector
            y: hp mass (background class for background)
            sample_weight: weights for events
        """
        self.fit(x)
        return self.transform(x, y, sample_weight)

class HpMTLBackgroundAugmenter():
    """ Class that assigns signal mass hypotheses to background training data with the PDF learned from signal for H+ multi-task learning. Each background event is copied for each of the signal hypotheses"""

    def __init__(self, backgroundclass=0, verbose=False):
        """ constructor
            backgroundclass: label for background
            verbose: print debug information if True
        """
        self.random_state=123456789
        self.backgroundclass=backgroundclass
        self.verbose=verbose
        self.hpmasses=[]
        self.sumweight=[]
        self.sumweightbackground=0.

    def fit(self, X, y, sample_weight):
        """ determines the mass PDF for events
            X: feature vector
            y: hp mass (background class for background)
            sample_weight: weights for events
        """    
        self.hpmasses=sorted(y.unique().tolist())
        self.hpmasses.remove(self.backgroundclass)
        for hpmass in self.hpmasses:
            self.sumweight.append(sample_weight[y==hpmass].sum())
        self.sumweightbackground=sample_weight[y==self.backgroundclass].sum()
        if self.verbose:
            print("Signal PDF:",self.hpmasses,self.sumweight)
            print("Background PDF:",self.sumweightbackground)


    def transform(self, X, y, sample_weight):
        """ augments background events with different signal labels according to fitted pdf
            X: feature vector
            y: hp mass (background class for background)
            sample_weight: weights for events
        """
        #first duplicate all signal
        arr_X=[X[y!=self.backgroundclass]]
        arr_y=[y[y!=self.backgroundclass]]
        arr_w=[sample_weight[y!=self.backgroundclass]]

        #now take care of background
        for i,hpmass in enumerate(self.hpmasses):
            Xnew=X[y==self.backgroundclass].copy()
            #ynew=y[y==self.backgroundclass].copy()
            ynew=pd.Series((-1)*hpmass, index=y[y==self.backgroundclass].index)
            Xnew.hpmass=ynew.abs()
            arr_X.append(Xnew)
            arr_y.append(ynew)
            arr_w.append(sample_weight[y==self.backgroundclass].copy().multiply(self.sumweight[i]/self.sumweightbackground))
    
        X=pd.concat(arr_X, axis=0)
        y=pd.concat(arr_y, axis=0)
        w=pd.concat(arr_w, axis=0)

        X, y, w = shuffle(X, y, w, random_state=self.random_state)

        if self.verbose:
            print("the following is the difference between + and - mass")
            print((w*((y>0)-0.5)*2).groupby(y.abs()).sum())
            print("the following is the sum of weights")
            print(w.groupby(X.hpmass).sum())            

        return X,y,w

