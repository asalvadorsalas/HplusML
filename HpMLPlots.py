"""Module with implementation of functions for plotting properties of input datasets and ML algorithms"""

import pandas as pd
import numpy as np
import HpMLUtils
import matplotlib.pyplot as plt
from math import sqrt

# pretty title for the standard variables in the feature matrix
variabletitles={
    "nJets" : "Number of jets",
    "nBTags_70": "Number of b-jets (70%)",
    "pT_jet1" : "Leading jet $p_T$ [GeV]",
    "Mbb_MindR_70" : "$M_{bb}$ (MindR, 70%) [GeV]",
    "pT_jet5":"5th leading jet $p_T$ [GeV]",
    "pT_jet1":"Leading jet $p_T$ [GeV]",
    "H1_all":"H1_all",
    "dRbb_avg_70":"$dR_{bb}$ (avg, 70%)",
    "dRlepbb_MindR_70":"$dR_{\ell bb}$ (MindR, 70%)",
    "Muu_MindR_70":"$M_{uu}$ (MindR, 70%) [GeV]",
    "HT_jets": "$H_T$ (jets) [GeV]",
    "Mbb_MaxPt_70" : "Mbb (MaxPt, 70%) [GeV]",
    "Mbb_MaxM_70" : "Mbb (MaxM, 70%) [GeV]",
    "Mjjj_MaxPt" : "M_jjj (MaxPt) [GeV]",
    "Centrality_all" : "Centrality (all)"
    }

def plotAverageVarianceAsFunctionOfMass(htf, variables=["Muu_MindR_70","nJets","Mbb_MindR_70","HT_jets",'H1_all','Mjjj_MaxPt','Mbb_MaxPt_70','dRbb_avg_70','pT_jet5', 'dRlepbb_MindR_70', 'nBTags_70','Mbb_MaxM_70', 'Centrality_all',"pT_jet1"], region="INC_ge6jge4b", hpmasses=[200,225,250,275,300,350,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000], save=False):
    """ plots the average and variance as a function of signal H+ mass
        htf: HpTrainingFrame object
        variables: list of feature names to consider for plots
        region: region of the analysis for which to plot
        hpmasses: list of H+ masses to consider for plot
        save: boolean, if true saves plot as png
    """
 
    averagebackground={}
    averagesignal={}
    sqrtvarupbackground={}
    sqrtvardownbackground={}
    sqrtvarupsignal={}
    sqrtvardownsignal ={}      

    for variable in variables:
        averagebackground[variable]=[]
        averagesignal[variable]=[]
        sqrtvarupbackground[variable]=[]
        sqrtvardownbackground[variable]=[]
        sqrtvarupsignal[variable]=[]
        sqrtvardownsignal[variable]=[]        

    scl=HpMLUtils.WeightedStandardScaler()
    msb=HpMLUtils.MultiSBWeightsScaler()

    for hpmass in hpmasses:
        X_train, X_test, X_eval, y_train, y_test,y_eval, w_train, w_test, w_eval=htf.prepare(region=region, hpmass=str(hpmass))
    
        msb.fit(X_train, y_train, sample_weight=w_train)
        X_train=msb.transform(X_train,y_train,w_train)
        scl.fit(X_train, sample_weight=w_train)
        X_train=scl.transform(X_train)
        df_train=pd.concat([X_train,y_train,w_train],axis=1)
        gb=df_train.groupby("process")
        for process, df in gb:
            wsum=np.sum(df.weight)
            for variable in variables:
                divisor=1.
                if "[GeV]" in variabletitles[variable]:
                    divisor=1000.
                ave=np.average(df[variable]/divisor,weights=df.weight)
                sqrtvar=HpMLUtils.sqrtvariance(df[variable]/divisor,weights=df.weight)
                print variable, hpmass, process, ave, sqrtvar, wsum
                if process==0:
                    averagebackground[variable].append(ave)
                    sqrtvarupbackground[variable].append(ave+sqrtvar)
                    sqrtvardownbackground[variable].append(ave-sqrtvar)
                else:
                    averagesignal[variable].append(ave)
                    sqrtvarupsignal[variable].append(ave+sqrtvar)
                    sqrtvardownsignal[variable].append(ave-sqrtvar)

    for variable in variables:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        pb=plt.plot(hpmasses, averagebackground[variable], 'k--')
        fb=plt.fill_between(hpmasses, sqrtvardownbackground[variable], sqrtvarupbackground[variable], alpha=0.2,facecolor="blue", label="background")
        ps=plt.plot(hpmasses, averagesignal[variable], 'k-')
        fs=plt.fill_between(hpmasses, sqrtvardownsignal[variable], sqrtvarupsignal[variable], alpha=0.2,facecolor="red", label="signal")
        #if modified:
        #    ax.set_ylim((-4, 4))
        plt.xlabel("m($H^+$) [GeV]")
        plt.ylabel(variabletitles[variable])
        plt.legend()
        plt.title("Average and variance")
        plt.tight_layout()
        if save:
            plt.savefig("averageandvarianceasfunctionofmass_"+region+"_"+variable+'.png')
    if save:
        plt.close('all')

def plotHistogram(classes, scores, weights, bins=[-2000,0,25,50,75,100,125,150,175,200,300,400,2000]):
    """ plots histogram of values (split by different classes)
        classes: Series of classes into which the histogram is split
        scores: values that are plotted (e.g. BDT score)
        weights: weights associated to the events
        bins: list of bins for the histogram
    """
    
    df=pd.concat([classes.reset_index(drop=True),pd.Series(scores),weights.reset_index(drop=True)],axis=1)
    df.columns=['class','score','weight']
    for name, values in df.groupby('class'):
        if name<=0:
            histtype="bar"
        else:
            histtype="step"
        sumw=values['weight'].sum()
        plt.hist(values['score'],bins=bins,histtype=histtype, weights=values['weight']/sumw, label=str(name))
    plt.legend()

def plotScoreHistogram(htf, method, hpmass="multi"):
    """ fits ML algorithm and plots score histogram (split by different classes)
        htf: HpTrainingFrame,
        method: machine learning algorithm
        hpmass: H+ mass for this plot
    """
    
    X_train, X_test, X_eval, y_train, y_test,y_eval, w_train, w_test, w_eval=htf.prepare(hpmass=hpmass)
    print "starting fit"
    method.fit(X_train, y_train, sample_weight=w_train)
    print "done with fit"
    y_pred=method.predict(X_test)
    plotHistogram(y_test,y_pred,w_test)
