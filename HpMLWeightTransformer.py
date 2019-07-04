"""Module with weight transformer functions for H+ machine learning algorithms"""

import numpy as np

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
        classes=sorted(np.unique(y))
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
        return X, y, sample_weight

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
        classes=sorted(np.unique(y))
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
        return X, y, sample_weight

class CustomWeightsScaler():
    """ Class that scales the weights of signal and background events to match the fraction from a weights dictionary"""

    def __init__(self, weights):
        """ constructor
            weights: dictionary of signal/background identifier and weight for the final dataset
        """

        sw=sum(weights.values())
        for key in weights.keys():
            weights[key]/=sw
        self.weights=weights
        self.scale_={}

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, 'scale_'):
            del self.scale_
    
    def fit(self,X,y, sample_weight):
        """Learns the sum of weights for all classes and calculates a scale factor for each class so that sum of weights for each category is equal to the requested weights
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        classes=sorted(y.unique())
        if classes!=sorted(self.weights.keys()):
            print("WARNING: weights identifiers",self.weights.keys()," different from data identifiers",classes)

        for classlabel in classes:
            sumweight=sample_weight[y==classlabel].sum()
            self.scale_[classlabel]=self.weights[classlabel]/sumweight
            
        return
        
    def transform(self, X, y, sample_weight, copy=None):
        """Transforms the sum of weights for all classes so that sum of weights for each category is equal to the requested weights
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        
        for classlabel in self.scale_:
            sample_weight[y==classlabel]*=self.scale_[classlabel]
        return X, y, sample_weight

class WeightsMultiplier():
    """ Class that scales the weights of signal and background events to match the fraction from a weights dictionary"""

    def __init__(self, scales, backgroundclass):
        """ constructor
            scales: dictionary of signal/background identifier and scale factor
            backgroundclass: label for background
        """
        self.backgroundclass=backgroundclass
        self.scales=scales
        
    def fit(self,X,y, sample_weight):
        """Fits a scale for the background so that the sum of weights for background remains 0.5 and for signal 0.5
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        
        #sumweight=sample_weight[y==self.backgroundclass].sum()
        #self.scale_[self.backgroundclass]=0.5/sumweight
        
        return
        
    def transform(self, X, y, sample_weight, copy=None):
        """Transforms the sum of weights for all classes so that weights for each category are multiplied by scales[category]
           X: feature matrix, ignored
           y: series of class labels
           sample_weight: Series of sample weights
        """
        
        for classlabel in self.scales:
            sample_weight[y==classlabel]*=self.scales[classlabel]
            
        return X, y, sample_weight