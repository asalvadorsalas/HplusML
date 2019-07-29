Tutorial
======
For the time being, [jupyter.pic.es](https://jupyter.pic.es/) does not have a working setup by *default*. Here are some steps to setup a local environment and install a kernel.
The kernel is the python instance used to run inside a notebook. If you run over scripts, you don't need ipykernel nor the last step.
**PLEASE** note that you only need to create the environment and the kernel installation **ONCE**. The environment can be activated again (step 2) and the kernel will appear as an option in the notebook 
1. Create local conda environment
···Conda is the friendly python environment that takes care of installing properly the compatible packages for you
···`conda create --prefix Hplus_tfgpu (prefix is to create it locally, if you use `--name` will be eliminated at the end of the session)
2. Activate it (you can deactivate it with `source deactivate`)
···While activated, you can install packages locally and recover the setup activating the environment again!
···`conda activate Hplus_tfgpu`
3. Install packages in environment
···Packages needed are tensorflow-gpu to run the gpu, pandas and tables to open the input, scikit-learn for BDTs and other tools, matplotlib for plotting tools and ipykernel to install the kernel.
···`conda install tensorflow-gpu` (requires confirmation)
···`pip install ipykernel pandas tables joblib scikit-learn matplotlib`
4. Install kernel with the created environment.
···This has to be done ONCE
···`python -m ipykernel install --user --name=Hplus_tfgpu` (or any other name)
5. Refresh the browser and make sure you see the new kernel option inside the jupyter nootebook or when creating a new one.

__if there is anything wrong with this setup tell me__
  

* Tutorial files are in here as they were too big to store in git. Copy the one locally (or more than one)
```
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b1024
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b128
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b2056
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b256
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b512
/nfs/at3/scratch/salvador/HplusML/tutorial/modelNN_b64
```
