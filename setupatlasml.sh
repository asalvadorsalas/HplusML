pip install tensorflow-gpu==1.12
now use pip install tensorflow-cpu==1.13.1
pip install joblib
pip install nbdime

#enable nbdiff for git
nbdime config-git --enable --global

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
pip install papermill

git config --global user.email "jglatzer@cern.ch"
git config --global user.name "Julian Glatzer"
