"""Base module for the default configuration of all employed ML algorithms"""

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

def getAdaBoostBDTClassifier(options={}):
    """the standard BDT classifer based on AdaBoost"""
    
    dt = DecisionTreeClassifier(criterion="gini",
                                max_depth=5,
                                min_samples_leaf=0.05,
                                random_state=0)
    bdt = AdaBoostClassifier(dt,
                             n_estimators=200,
                             learning_rate=0.13,
                             algorithm='SAMME',
                             random_state=0)
    bdt.set_params(options={})
    return bdt

def getGradientBDTClassifier(options={}):
    """the standard BDT classifier based on Gradient Boosting"""
    
    bdt = GradientBoostingClassifier(n_estimators=120,
                                     learning_rate=0.13,
                                     max_depth=5,
                                     min_weight_fraction_leaf=0.01,
                                     random_state=0)
    bdt.set_params(**options)
    return bdt

def getAdaBoostBDTRegressor(options={}):
    """the standard BDT regressor based on AdaBoost"""
    
    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                               n_estimators=500, learning_rate=0.13, loss='ls')
    return clf

def getGradientBDTRegressor(options={}):
    """the standard BDT regressor based on Gradient Boosting"""
    
    params = {'n_estimators': 20, 'max_depth': 5, 'min_samples_split': 2,
              'learning_rate': 0.13, 'loss': 'ls'}
    clf = GradientBoostingRegressor(**params)
    return clf




