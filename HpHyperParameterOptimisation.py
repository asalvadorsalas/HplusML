"""Module for the optimisation of Hyperparameters for a wide range of ML algorithms"""

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import roc_curve,roc_auc_score, auc
import numpy as np
import matplotlib.pyplot as plt
import HpMLUtils

class HpOptimise():
    """Class with useful functions for the optimisation of hyperparameters for a wide range of ML algorithms"""

    def __init__(self, name, method, X_train, y_train, w_train, X_test, y_test, w_test):
        """ constructor
            name: string, name of the method, used for saving images,...
            method: classifier, e.g. GradientBoostingClassifier with fit and predict method, some function require a (staged) decision function
            X_{train,test): feature matrix for training/testing data
            y_{train,test): class labels for training/testing data
            w_{train,test): sample weights for training/testing data
        """
        
        self.name=name
        self.method=method
        self.X_train=X_train
        self.y_train=y_train
        self.w_train=w_train
        self.X_test=X_test
        self.y_test=y_test
        self.w_test=w_test

    def test(self, X_train, y_train, w_train, X_test, y_test, w_test, stagedresults=False):
        """ returns (staged) ROC AUC value for testing and training dataset
            X_{train,test): feature matrix for training/testing data
            y_{train,test): class labels for training/testing data
            w_{train,test): sample weights for training/testing data
            stagedresults: bool, if true returns training and testing ROC AUC values as a function of the number of boosting iterations
        """
        
        if self.method._estimator_type=="classifier":
            if stagedresults:
                test = np.empty(len(self.method.estimators_))
                for i, pred in enumerate(self.method.staged_decision_function(X_test)):
                    test[i]=1.-roc_auc_score(y_test, pred, sample_weight=w_test)
                train = np.empty(len(self.method.estimators_))
                for i, pred in enumerate(self.method.staged_decision_function(X_train)):
                    train[i]=1.-roc_auc_score(y_train, pred, sample_weight=w_train)
            else:
                test=1.-roc_auc_score(y_test, self.method.decision_function(X_test), sample_weight=w_test)
                train=1.-roc_auc_score(y_train,self.method.decision_function(X_train), sample_weight=w_train)

        elif self.method._estimator_type=="regressor": #regressor
            if stagedresults:
                test = np.empty(len(self.method.estimators_))
                for i, pred in enumerate(self.method.staged_predict(X_test)):
                    test[i]=1.-roc_auc_score(y_test, pred, sample_weight=w_test)
                train = np.empty(len(self.method.estimators_))
                for i, pred in enumerate(self.method.staged_predict(X_train)):
                    train[i]=1.-roc_auc_score(y_train, pred, sample_weight=w_train)
            else:
                test=1.-roc_auc_score(y_test, self.method.predict(X_test), sample_weight=w_test)
                train=1.-roc_auc_score(y_train,self.method.predict(X_train), sample_weight=w_train)
        else:
            print "Unknown ML algorithm, neither regressor nor classifier"
        return test, train

    def trainAndTest(self, stagedresults=False, options={}, silent=False):
        """ fits ML classifier on training data and return evaluation metric (ROC AUC) for training and testing data
            stagedresults: bool, if true returns training and testing ROC AUC values as a function of the number of boosting iterations
            options: parameters for the classifier
            silent: bool, if true no status information is printed to the command line
        """
        
        self.method.set_params(**options)
        if not silent:
            print "starting training"
        self.method.fit(self.X_train, self.y_train, sample_weight = self.w_train)
        if not silent:
            print "training done"
        return self.test(stagedresults=stagedresults, X_train=self.X_train, y_train=self.y_train, w_train=self.w_train, X_test=self.X_test, y_test=self.y_test, w_test=self.w_test)

    def getDefaultParams(self):
        """ return default parameters for different ML algorithms"""
        
        if isinstance(self.method, AdaBoostClassifier):
            return {
                'n_estimators': 200,
                'learning_rate': 0.13,
                'base_estimator__max_depth': 5
            }
        if isinstance(self.method, GradientBoostingClassifier):
            return {
                'n_estimators': 200,
                'learning_rate': 0.13,
                'max_depth': 5
            }
        if isinstance(self.method, AdaBoostRegressor):
            return {
                'n_estimators': 50,
                'learning_rate': 0.13,
                'base_estimator__max_depth': 5
            }
        if isinstance(self.method, GradientBoostingRegressor):
            return {
                'n_estimators': 20,
                'learning_rate': 0.13,
                'max_depth': 5
            }

    def getParamGrid(self):
        """ return default parameter grid for different ML algorithms for validation curves and random search of hyperparameters"""
        
        if isinstance(self.method, AdaBoostClassifier):
            return {'n_estimators': [50,120,200,400,800],
                    'learning_rate': [0.05,0.1,0.13,0.2,0.5],
                    'base_estimator__max_depth': [3,4,5,6]
            }
        if isinstance(self.method, GradientBoostingClassifier):
            return {'n_estimators': [50,120,200,400,800],
                    'learning_rate': [0.05,0.1,0.13,0.2,0.5],
                    'max_depth': [3,4,5,6]
            }

    def drawMultiClassROCCurve(self, backgroundclass=0, nlines=6):
        """Compute and draw ROC curves for a regressor trained to identify several signal hypotheses (+area under the curve)
           backgroundclass: object, class label for background
           nlines: integer, number of ROC curves shown (for many signal hypothesis not all ROC curves will be shown)
        """
        
        y_pred=self.method.predict(self.X_test)
        classes=np.sort(self.y_test.unique())
        #remove the background from the classes
        classes=np.delete(classes,np.where(classes==backgroundclass))

        #select nline equally space lines
        classes=classes[np.linspace(0,len(classes)-1,nlines).astype(int)]

        for signalclass in classes:
            mask=(self.y_test==signalclass) | (self.y_test==backgroundclass)
            fpr, tpr, thresholds = roc_curve(self.y_test[mask]==signalclass, y_pred[mask], sample_weight = self.w_test[mask])

            roc_auc = auc(fpr, tpr, reorder=True)

            plt.plot(fpr, tpr, lw=1, label='ROC %s (area = %0.2f)'%(signalclass, roc_auc))
        
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.grid()

    def drawROCCurve(self):
        """Compute and draw ROC curve (+area under the curve)
           requires decision_function for ML algorithm
        """

        fpr, tpr, thresholds = roc_curve(self.y_test, self.method.decision_function(self.X_test), sample_weight = self.w_test)

        roc_auc = auc(fpr, tpr, reorder=True)

        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
        
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        #plt.show()
        
    def saveROCCurve(self, filename=None, fit=False):
        """calculate, draw and save ROC curve
           filename: None or string, filename under which the plot will be saved (default: None, auto-generated file name)
           fit: bool if True, method will be fit to training data before evaluation
        """
        
        title=self.name+" "+str(self.method.get_params())
        if fit:
            print "starting training"
            self.method.fit(self.X_train, self.y_train, sample_weight = self.w_train)
            print "training done"
        self.drawROCCurve(title=title)
        if filename is None:
            filename="roccurve_"+type(self.method).__name__+"_"+self.name
            for key in sorted(self.method.get_params().keys()):
                filename=filename+"_"+key+str(self.method.get_params()[key]).replace(".","")
            filename+=".png"
        print "Saving plot as ", filename
        plt.savefig(filename)
        
    def calculateLearningCurve(self,train_sizes=np.linspace(.1, 1.0, 10), category="all", writetotxt=False):
        """Compute learning curve (i.e. evaluation metric 1-ROC AUC as a function of the training size) for ML algorithm
           train_sizes: np array of floats, array of values for fractional training considered for learning curve (default: 10%-100% in steps of 10%)
           category: category label of events to be sample from (i.e. category=0 takes all signal events and samples only background events), default: all (i.e. sample from signal and background)
           writetotxt: bool, if true results are written to text file (default: false)
        """

        if writetotxt:
            txtfilename="learningcurve_"+type(self.method).__name__+"_"+self.name+".txt"
            txtfile = open(txtfilename,"a")
            txtfile.write("#Learning curve for name="+self.name+" "+str(self.method.get_params())+"\n")
            txtfile.write("#fraction of the training size, ROC(test), ROC(train)\n")

        train_scores=[]
        test_scores=[]
        for fraction_train_size in train_sizes:
            if category=="all":
                X_train,y_train,w_train=HpMLUtils.sample(self.X_train, self.y_train,self.w_train, frac=fraction_train_size)
            else:
                X_train,y_train,w_train=HpMLUtils.sample(self.X_train, self.y_train,self.w_train, frac=fraction_train_size, categories=self.y_train, categorytosample=category)
            self.method.fit(X_train, y_train, sample_weight = w_train)
            rocvalue_test,rocvalue_train=self.test(X_train=X_train, y_train=y_train, w_train=w_train, X_test=self.X_test, y_test=self.y_test, w_test=self.w_test)
            
            if writetotxt:
                txtfile.write(str(fraction_train_size)+" "+str(rocvalue_test)+" "+str(rocvalue_train)+"\n")

            train_scores.append(rocvalue_train)
            test_scores.append(rocvalue_test)

        return train_sizes, train_scores, test_scores

    def drawLearningCurve(self, title, train_sizes=None, train_scores=None, test_scores=None, txtfilename=None):
        """Draw learning curve (i.e. evaluation metric as a function of the training size) for ML algorithm
           train_sizes: np array of floats, array of values for fractional training considered for learning curve (default: 10%-100% in steps of 10%)
           train/test_scores: training/testing score for the different fractional training sizes (as e.g. calculated by calculateLearningCurve())
           txtfilename: optional text frile from which the results are read (default: None=not used)
        """
        
        if not txtfilename is None:
            if txtfilename=="":
                txtfilename="learningcurve_"+type(self.method).__name__+"_"+self.name+".txt"

            txtfile = open(txtfilename,"r") 
            parammap={}
            train_sizes=[]
            test_scores=[]
            train_scores=[]
            for line in txtfile:
                if "Learning curve for" in line:
                    for itm in line.strip().split(" "):
                        if "=" in itm:
                            param=itm.split("=")
                            parammap[param[0]]=param[1]
                if not "#" in line:
                    arr=[float(i) for i in line.strip().split(" ")]
                    train_sizes.append(arr[0])
                    train_scores.append(arr[2])
                    test_scores.append(arr[1])
                else:
                    print line

        plt.figure()
        plt.title(title, fontsize=10)
        plt.plot(train_sizes, train_scores, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores, 'o-', color="g",
                 label="Cross-validation score")
        plt.xlabel('Fractional size of training sample')
        plt.ylabel("1-AUC")
        plt.legend(loc="upper right")
        plt.grid()
    
    def saveLearningCurve(self, train_sizes=np.linspace(.1, 1.0, 10), category="all", filename=None):
        """Compute, draw and save learning curve (i.e. evaluation metric 1-ROC AUC as a function of the training size) for ML algorithm
           train_sizes: np array of floats, array of values for fractional training considered for learning curve (default: 10%-100% in steps of 10%)
           category: category label of events to be sample from (i.e. category=0 takes all signal events and samples only background events), default: all (i.e. sample from signal and background)
           filename: None or string, filename under which the plot will be saved (default: None, auto-generated file name)
        """
        
        _, train_scores, test_scores = self.calculateLearningCurve(train_sizes=train_sizes, category=category)
        title=str(type(self.method))+"_"+self.name
        self.drawLearningCurve(title=title, train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores)
        if filename is None:
            filename="learningcurve_"+type(self.method).__name__+"_"+self.name
            for key in sorted(self.method.get_params().keys()):
                filename=filename+"_"+key+str(self.method.get_params()[key]).replace(".","")
            filename+=".png"
        print "Saving plot as ", filename
        plt.savefig(filename)
        
    def calculateValidationCurve(self, variable="learning_rate", values=None, writetotxt=False):
        """Calculate validation curve (i.e. evaluation metric 1-ROC AUC as a function of the BDT boosting iteration for different hyperparameters) for ML algorithm
           variable: string, name of the hyperparameter which is varied
           values: list of values for the hyperparameter
           writetotxt: bool, if true results are written to text file (default: false)
        """

        if writetotxt:
            txtfilename="validationcurve_"+type(self.method).__name__+"_"+self.name+"_"+variable+".txt"
            txtfile = open(txtfilename,"a") 
            txtfile.write("#Validation curve for variable="+variable+" name="+self.name+" "+str(self.method.get_params())+"\n")
            txtfile.write("#value ROC(test), ROC(train)\n")

        if values==None:
            values=self.getParamGrid()[variable]
        
        train_scores={}
        test_scores={}
        saved_options=self.method.get_params()

        #from joblib import Parallel, delayed
        #import multiprocessing
        #num_cores = multiprocessing.cpu_count()
        def get_scores(value):
            options=saved_options
            options[variable]=value
            print "Training for ", value
            if writetotxt:
                txtfile.write("Classifier name="+self.name+" "+str(options)+"\n")
            test,train=self.trainAndTest(stagedresults=True, silent=True, options=options)
            if writetotxt:
                txtfile.write("testscore "+' '.join(map(str,test))+"\n")
                txtfile.write("trainscore "+' '.join(map(str,train))+"\n") 
            return train,test
        rt=[get_scores(value) for value in values]
        #rt=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(get_scores)(value) for value in values)
        return dict(zip(values,[itm[0] for itm in rt])), dict(zip(values,[itm[1] for itm in rt]))

    def drawValidationCurve(self, variable, values, train_scores, test_scores, showtrain=False, txtfilename=None):
        """Draw validation curve (i.e. evaluation metric 1-ROC AUC as a function of the BDT boosting iteration for different hyperparameters) for ML algorithm
           variable: string, name of the hyperparameter which is varied
           values: list of values for the hyperparameter
           train/test_scores: training/testing score for the different fractional training sizes (as e.g. calculated by calculateLearningCurve())
           showtrain: bool, if true training scores will be shown on the plot
           txtfilename: optional text frile from which the results are read (default: None=not used)
        """
                          
        ntrees={}
        if txtfilename is not None:
            if txtfilename=="":
                txtfilename="validationcurve_"+type(self.method).__name__+"_"+self.name+"_"+variable
                for key in sorted(options.keys()):
                    txtfilename=filename+"_"+key+str(options[key]).replace(".","")
                txtfilename+=".txt"
            txtfile = open(txtfilename,"r") 
            parammap={}
            test_scores={}
            train_scores={}
            value=-99999
            for line in txtfile:
                if "Validation curve for" in line:
                    for itm in line.strip().split(" "):
                        if "=" in itm:
                            param=itm.split("=")
                            parammap[param[0]]=param[1]
                if not "#" in line and parammap["variable"]==variable:
                    if "Classifier" in line:
                        for itm in line.strip().split(" "):
                            if "=" in itm:
                                param=itm.split("=")
                                if param[0]==variable:
                                    value=param[1]
                    
                    if "testscore" in line:
                        arr=[float(i) for i in line.strip().split(" ")[1:]]
                        ntrees[value]=range(1,len(arr)+1)
                        test_scores[value]=arr
                    if "trainscore" in line:
                        arr=[float(i) for i in line.strip().split(" ")[1:]]
                        train_scores[value]=arr
        else:
            for value in values:
                ntrees[value]=range(1,len(train_scores[value])+1)

        plt.figure()
        title=str(type(self.method))+"_"+self.name
        plt.title(title, fontsize=8)
        
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(ntrees))))

        for label in sorted(ntrees.keys()):
            c=next(color)
            if showtrain:
                plt.plot(ntrees[label], train_scores[label], '--', color=c,
                         label=variable+"="+str(label)+" train")
            plt.plot(ntrees[label], test_scores[label], '-', color=c,
                     label=variable+"="+str(label)+" test")
        plt.xlabel('Number of boosting iterations')
        plt.ylabel("1-AUC")
        plt.legend(loc="upper right")
        plt.grid()
    
    def saveValidationCurve(self, variable="learning_rate", values=None, filename=None):
        """Compute, draw and save validation curve (i.e. evaluation metric 1-ROC AUC as a function of the BDT boosting iteration for different hyperparameters) for ML algorithm
           variable: string, name of the hyperparameter which is varied
           values: list of values for the hyperparameter
           filename: None or string, filename under which the plot will be saved (default: None, auto-generated file name)
        """
                        
        if values==None:
            values=self.getParamGrid()[variable]
        
        train_scores, test_scores = self.calculateValidationCurve(variable=variable, values=values)
        self.drawValidationCurve(variable=variable, values=values, train_scores=train_scores, test_scores=test_scores,showtrain=len(train_scores)<=2)
        if filename is None:
            filename="validationcurve_"+type(self.method).__name__+"_"+self.name+"_"+variable
            for key in sorted(self.method.get_params().keys()):
                filename=filename+"_"+key+str(self.method.get_params()[key]).replace(".","")
            filename+=".png"
                          
        print "Saving plot as ", filename
        plt.savefig(filename)
        
    def saveAllValidationCurves(self):
        """Compute, draw and save validation curves (i.e. evaluation metric 1-ROC AUC as a function of the BDT boosting iteration for different hyperparameters) for all default hyperparameters"""
                          
        variables=self.getParamGrid().keys()
        for variable in variables:
            if variable!="n_estimators":
                print "Validation curve for", variable
                self.saveValidationCurve(variable=variable)
    
    def randomSearch(self):
        pass
