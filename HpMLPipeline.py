"""Module which implements a SKLearn transformer pipeline with sample weights"""

from __future__ import print_function
from sklearn.utils.metaestimators import _BaseComposition

class PipelineWithWeights(_BaseComposition):
    """ Transformer pipeline that correctly uses sample weights (the standard SKLearn pipeline does not do this correctly)"""

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]

        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

    def fit(self, X, y, sample_weight):
        Xtmp=X.copy()
        ytmp=y.copy()
        wtmp=sample_weight.copy()
        for name, step in self.steps:
            step.fit(Xtmp,ytmp, wtmp)
            Xtmp,ytmp,wtmp=step.transform(Xtmp,ytmp,sample_weight=wtmp)
    
    def transform(self, X, y, sample_weight):
        for name, step in self.steps:
            X,y,sample_weight=step.transform(X,y,sample_weight=sample_weight)
        return X,y,sample_weight
