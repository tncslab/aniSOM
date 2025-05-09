#!/bin/bash
# run all the lorenz  python scriptfiles in the folder

# create a manual progress-bar for the 9 items
ZERO='[---------] (0/9)'
ONE='[#--------] (1/9)'
TWO='[##-------] (2/9)'
THREE='[###------] (3/9)'
FOUR='[####-----] (4/9)'
FIVE='[#####----] (5/9)'
SIX='[######---] (6/9)'
SEVEN='[#######--] (7/9)'
EIGHT='[########-] (8/9)'
NINE='[#########] (9/9)'

# define a function it has one arg that is a string and it is clearing the screen and echoing the string
function pbar {
    clear
    echo $1
}




# Run the files

# PCA
conda activate maco_rev1
python 00_pca.py
conda deactivate

pbar $ONE


# ICA
conda activate maco_rev1
python 01_ica.py
conda deactivate

pbar $TWO


# CCA
conda activate maco_rev1
python 02_cca.py
conda deactivate

pbar $THREE

# DCA
conda activate dca
python 03_dca.py
conda deactivate

pbar $FOUR

# sfa
conda activate fsa2
python 04_sfa.py
conda deactivate

pbar $FIVE

# DCCA
conda activate dcca_env
python3 05_dcca.py
conda deactivate

pbar $SIX

# Random Control
conda activate maco_rev1
python 06_random.py
conda deactivate

pbar $SEVEN


# Sh-Rec
conda activate shrec
python 07_shrec.py
conda deactivate


pbar $EIGHT


# AniSOM
conda activate maco_rev1
python 08_anisom.py
conda deactivate

pbar $NINE

# Combine the results
conda activate maco_rev1
python Z_combine_final_res.py
conda deactivate





