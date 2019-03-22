"""Module with several helper functions to build, train and evaluate H+ Keras models"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class RocCallback(Callback):
    """ A Keras callback that calculates the ROC AUC"""

    def __init__(self,training_data,validation_data, verbose=True):
        self.verbose=verbose
        self.x = training_data[0]
        self.y = training_data[1]
        self.w = training_data[2]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.w_val = validation_data[2]
        self.roc=[]
        self.roc_val=[]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred, sample_weight=self.w)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val, sample_weight=self.w_val)
        if self.verbose:
            print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))))
        self.roc_val.append(roc_val)
        self.roc.append(roc)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class HpFeedForwardModel():
    """ A simple feed forward NN based on Keras"""

    def __init__(self, configuration, l2threshold=None, dropout=None, input_dim=15, verbose=True):
        """ constructor
        configuration: list of the number of nodes per layer, each item is a layer
        l2threshold: if not None a L2 weight regularizer with threshold <l2threshold> is added to each leayer
        dropout: if not None a dropout fraction of <dropout> is added after each internal layer
        input_dim: size of the training input data
        verbose: if true the model summary is printed
        """
        
        self.callbacks = []
        self.configuration=configuration
        self.dropout=dropout
        self.l2threshold=l2threshold
        self.model = Sequential()
        for i,layer in enumerate(configuration):
            if i==0:
                if l2threshold==None:
                    self.model.add(Dense(layer, input_dim=input_dim, activation='relu'))    
                else:
                    self.model.add(Dense(layer, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(l2threshold)))    
            else:
                if l2threshold==None:
                    self.model.add(Dense(layer, activation='relu'))
                else:
                    self.model.add(Dense(layer, activation='relu', kernel_regularizer=regularizers.l2(l2threshold)))
            if dropout!=None:
                self.model.add(Dropout(rate=dropout))
        #final layer is a sigmoid for classification
        self.model.add(Dense(1, activation='sigmoid'))
        #model.add(Dense(5, activation='relu'))

        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        if verbose:
            self.model.summary()
    
    def train(self, trainData, testData, epochs=100, patience=15, callbacks=None):
        """ train the Keras model with Early stopping, will return test and training ROC AUC
        trainData: tuple of (X_train, y_train, w_train)
        trainData: tuple of (X_test, y_test, w_test)
        epochs: maximum number of epochs for training
        patience: patience for Early stopping based on validation loss
        callbacks: 
        """

        X_train=trainData[0]
        y_train=trainData[1]
        w_train=trainData[2]
        X_test=testData[0]
        y_test=testData[1]
        w_test=testData[2]

        if callbacks is None:
            self.callbacks.append(EarlyStopping(monitor='val_loss', 
                                                patience=patience))
            self.callbacks.append(ModelCheckpoint(filepath='model_nn_'+str(self.configuration)+"_dropout"+str(self.dropout)+"_l2threshold"+str(self.l2threshold)+".hdf5", 
                                                  monitor='val_loss',
                                                  save_best_only=True))
            self.callbacks.append(RocCallback(training_data=trainData,validation_data=testData))
        else:
            self.callbacks=callbacks

        self.history=self.model.fit(X_train,y_train, sample_weight=w_train,
                                    batch_size=50, epochs=epochs, callbacks=self.callbacks,
                                    validation_data=testData)

        self.model.load_weights("model_nn_"+str(self.configuration)+"_dropout"+str(self.dropout)+"_l2threshold"+str(self.l2threshold)+".hdf5")
        y_pred_test=self.model.predict(X_test).ravel()
        y_pred_train=self.model.predict(X_train).ravel()
        roc_test =roc_auc_score(y_test,  y_pred_test,  sample_weight=w_test)
        roc_train=roc_auc_score(y_train, y_pred_train, sample_weight=w_train)
        #print(self.configuration, roc_test, roc_train)
        
        return roc_test, roc_train

    def plotTrainingValidation(self):
        """draws plots for loss, binary accuracy and ROC AUC"""

        loss_values=self.history.history['loss']
        val_loss_values=self.history.history['val_loss']
        acc_values=self.history.history['binary_accuracy']
        val_acc_values=self.history.history['val_binary_accuracy']

        rocauc_values=None
        val_rocauc_values=None
        bestepoch=None
        for cb in self.callbacks:
            if hasattr(cb, 'roc') and hasattr(cb, 'roc_val'):
                rocauc_values=cb.roc
                val_rocauc_values=cb.roc_val
            if hasattr(cb, 'stopped_epoch') and hasattr(cb, 'patience'):
                bestepoch=cb.stopped_epoch-cb.patience+1
  
        epochs=range(1,len(acc_values)+1)
        plt.figure()
        plt.plot(epochs, loss_values, "bo",label="Training loss")
        plt.plot(epochs, val_loss_values, "b",label="Validation loss")
        plt.legend(loc=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if not bestepoch is None:
            plt.axvline(x=bestepoch)

        ax=plt.figure()
        plt.plot(epochs, acc_values, "bo",label="Training acc")
        plt.plot(epochs, val_acc_values, "b",label="Validation acc")
        plt.legend(loc=0)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        if not bestepoch is None:
            plt.axvline(x=bestepoch)

        if not rocauc_values is None:
            ax=plt.figure()
            plt.plot(epochs, rocauc_values, "bo",label="Training ROC AUC")
            plt.plot(epochs, val_rocauc_values, "b",label="Validation ROC AUC")
            plt.legend(loc=0)
            plt.xlabel("Epochs")
            plt.ylabel("ROC AUC")
            if not bestepoch is None:
                plt.axvline(x=bestepoch)
