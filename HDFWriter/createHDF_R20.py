#do not forget to setup hdf with setupATLAS; lsetup hdf5; requires python module tables
from __future__ import print_function
from HpData import *
from pandas import HDFStore

path="/nfs/at3/scratch/jglatzer/L2Output_R20_Nov_v2_pandaspython3"

btagWP=70
hpanalysis=HpAnalysis(20,btagWP,"fitconfigR20.json")
#hpanalysis.readData(['data'])
hpanalysis.readData() #regions=['INC_ge6jge4b'], samples=["Hp300"])
#hpanalysis.readData(regions=['INC_ge6jge4b'], samples=["Hp300"])
df_mc=hpanalysis.df_mc.drop("jet_pt", axis=1)

hdf =HDFStore(path+"/pandas_allregions.h5")

df_mc.to_hdf(hdf,"allregions")
print(hdf.keys(), hpanalysis.df_mc.shape)
hdf.close()

for region in hpanalysis.df_mc.region.unique():
    print(region)
    hdf =HDFStore(path+"/pandas_"+region+".h5")
    df=df_mc[hpanalysis.df_mc.region==region]
    df.to_hdf(hdf,region)
    print(hdf.keys(), df.shape)
    hdf.close()

