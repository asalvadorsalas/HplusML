"""Module to simplify H+ data import """
from __future__ import print_function
from IPython.core.debugger import set_trace
import pandas as pd
from root_pandas import read_root
import json
import os.path
import re
import numpy
import sys

#df.apply(setRegionR21,args=(self.btagWP,), axis=1)
def getR21TTHWeight(event):
    year=1516
    if event["randomRunNumber"]>0 or  event["mujets_2017_MV2c10"]>0:
        return 1
    return 0

class HpAnalysis:
    """Class that provides the H+ analysis functions"""

    def __init__(self, release,btagWP,fitconfigfilename):
        """Constructor which sets the release (20 or 21) and b-tagging working point"""

        if release!=20 and release!=21:
            print("ERROR: Unknown release ", release, " using R21 now.")
            self.release=21
        else:
            self.release=release

        if btagWP!=60 and btagWP!=70 and btagWP!=77 and btagWP!=80:
            print("ERROR: Unknown btagging WP ", btagWP, " using 70% now.")
            self.btagWP=70
        else:
            self.btagWP=btagWP
        
        with open(fitconfigfilename) as f:
            self.fitconfig = json.load(f)
        
        self.feature_names=["nJets","nBTags_70","jet_pt","Mbb_MindR_70","pT_jet5","H1_all","dRbb_avg_70","dRlepbb_MindR_70","Muu_MindR_70","HT_jets","Mbb_MaxPt_70","Mbb_MaxM_70","Mjjj_MaxPt","Centrality_all"]

        for item in [225,250,275,300,350,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000]:
            self.feature_names.append("HpNN_"+str(item))
            if item in [350,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000]:
                self.feature_names.append("HpBDT_Semilep_v1_INC_BDT"+str(item))
            elif item in [225,250,275,300]:
                self.feature_names.append("HpBDT_Semilep_v2_HF_BDT"+str(item))
            else:
                print ("ERROR in BDT variable loading")
                sys.exit()
        for item in [225,250,275,300,600,1200,1800]:
            self.feature_names.append("HpDiscriminant_"+str(item))

        self.feature_names.append("randomRunNumber")
        self.getGeneralSettings()
        self.df_mc=None
        self.df_data=None

    def get1DFeatureNames(self):
        return ["pT_jet1" if i == "jet_pt" else i for i in self.feature_names]

    def regions(self):
        return [region["Name"] for region in self.fitconfig["Region"]]

    def getGeneralSettings(self):
        if not "Job" in self.fitconfig:
            print("ERROR: Did not find Job in fit configuration")
        if not "NtuplePaths" in self.fitconfig["Job"]:
            print("ERROR: Did not find NtuplePaths in fit configuration -> Job")
        self.inputpath=self.fitconfig["Job"]["NtuplePaths"]
        if not "NtupleName" in self.fitconfig["Job"]:
            print("ERROR: Did not find NtupleName in fit configuration -> Job")
        self.treename=self.fitconfig["Job"]["NtupleName"]
        if not "MCweight" in self.fitconfig["Job"]:
            print("ERROR: Did not find MCweight in fit configuration -> Job")
        self.mcweight=self.fitconfig["Job"]["MCweight"]
        self.lumiscale=1.

    def readData(self, samples=[], regions=[]):
        if len(samples)>1 and samples[0]!='data':
            self.df_mc=None

        for sample in self.fitconfig["Sample"]:
            if "fake" in sample['Name']:
                continue
            if ((len(samples)==0 and sample['Name']!='data') or sample['Name'] in samples)  and sample['Type'].upper()!="GHOST":
                if sample['Name']=='data':
                    self.df_data=None
                for region in self.fitconfig["Region"]:
                    if len(regions)==0 or region["Name"] in regions:
                        print("Reading",sample['Name'],"in region", region["Name"])
                        selection=region["Selection"]
                        if "Selection" in sample:
                            selection="("+selection+") && ("+sample["Selection"]+")"

                        mcweight=""
                        if sample['Name']!='data':
                            mcweight=self.mcweight
                            if "MCweight" in region:
                                mcweight="("+self.mcweight+")*("+region["MCweight"]+")"
                            if "MCweight" in sample:
                                if "NormalizedByTheory" in sample:
                                    if sample["NormalizedByTheory"]:
                                        mcweight="("+self.mcweight+")*("+sample["MCweight"]+")"
                                        if "LumiScale" in sample:
                                            self.lumiscale=sample["LumiScale"]
                                        else:
                                            print("ERROR: Did not find LumiScale for sample ", sample["Name"], " even though it is normalized by theory")
                                            self.lumiscale=1.
                                    else:
                                        mcweight=sample["MCweight"]
                                        lumiscale=1.
                                else:
                                    mcweight="("+self.mcweight+")*("+sample["MCweight"]+")"
                                    if "LumiScale" in sample:
                                        self.lumiscale=sample["LumiScale"]
                                    else:
                                        print("ERROR: Did not find LumiScale for sample ", sample["Name"], " even though it is normalized by theory")
                                        self.lumiscale=1.

                        #get the list of input files
                        datafiles=[]
                        if not "NtuplePathSuffs" in region and "NtuplePathSuff" in region:
                            region["NtuplePathSuffs"]=region["NtuplePathSuff"]
                        if type(region["NtuplePathSuffs"])!=list:
                            region["NtuplePathSuffs"]=[region["NtuplePathSuffs"]]
                        for ntuplepath in region["NtuplePathSuffs"]:
                            if not "NtupleFiles" in sample and "NtupleFile" in sample:
                                sample["NtupleFiles"]=sample["NtupleFile"]
                            if type(sample["NtupleFiles"])!=list:
                                sample["NtupleFiles"]=[sample["NtupleFiles"]]
                            for ntuplefile in sample["NtupleFiles"]:
                                datafile=self.inputpath.rstrip("/")+"/"+ntuplepath.strip("/")+"/"+ntuplefile.lstrip("/")+".root"
                                if os.path.isfile(os.path.expanduser(datafile)):
                                    datafiles.append(datafile)

                        #get the data frame from the root file
                        columns=self.feature_names+["eventNumber"]
                        if self.lumiscale!=1.:
                            mcweight=mcweight+"*"+str(self.lumiscale)
                        #if True:
                        #    print "Resetting the weight to the TMVA weight"
                        #    mcweight="fabs(weight_leptonSF * weight_bTagSF_"+str(self.btagWP)+" * weight_mc * weight_pileup * weight_jvt * weight_normalise * weight_ttbb_Norm * weight_ttbb_Shape_SherpaNominal)"
                        if sample['Name']!='data':
                            columns=columns+["noexpand:"+mcweight]                            
                        
                        print("Reading:", datafiles, self.treename, columns, selection)
                        tmpdf = read_root(datafiles,self.treename,columns=columns,where=selection)
                        tmpdf.rename(columns={mcweight: 'weight'}, inplace=True) #remove the ugly name of the mcweight column
                        if "weight" in tmpdf.columns: #special case for mc
                            tmpdf["weight"]=tmpdf["weight"].apply(lambda x: x[0] if type(x)==numpy.ndarray else x)
                        tmpdf["process"]=str(sample['Name'])
                        if not "Group" in sample:
                            tmpdf["group"]=sample['Name']
                        else:
                            tmpdf["group"]=sample['Group']
                        tmpdf["region"]=region["Name"]
                        if sample['Name']=="data":
                            tmpdf['weight']=1.

                        if sample['Name']=='data':
                            if type(self.df_data)==None:
                                self.df_data=tmpdf
                            else:
                                self.df_data=pd.concat([self.df_data,tmpdf],axis=0)
                        else:
                            #tmpdf=tmpdf[tmpdf['weight'] != 0] #ATTENTION: this might be dangerous for cut flows
                            if type(self.df_data)==None:
                                self.df_mc=tmpdf
                            else:
                                self.df_mc=pd.concat([self.df_mc,tmpdf],axis=0)
                                
        print("adding jet_pt1 column")
        if self.df_data is not None:
            self.df_data.loc[:, 'pT_jet1'] = self.df_data.jet_pt.map(lambda x: x[0])
            self.df_data.reset_index(inplace = True) 
            self.df_data["hpmass"]=-2

        if self.df_mc is not None:
            self.df_mc.loc[:, 'pT_jet1'] = self.df_mc.jet_pt.map(lambda x: x[0])
            self.df_mc.reset_index(inplace = True) 
            self.df_mc["hpmass"]=self.df_mc.apply(lambda x: int(x['process'].replace("Hp","")) if "Hp" in x['process'] else -1,axis=1)
            self.df_mc.loc[:, 'year'] = self.df_mc.randomRunNumber.map(lambda x: "mc16a" if x<=311481 else "mc16d" if x <=340453 else "mc16e") 
            self.df_mc.drop("randomRunNumber",axis=1)
