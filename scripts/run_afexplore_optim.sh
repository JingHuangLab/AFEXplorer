#!/bin/bash

feat_dir=$1 # Directory of features.pkl and output
target_state=$2 # 'kinase', 'IOMemP', 'ADK'
protein_type=$3 # "in" or "out"
afparam_dir=$4 # Directory of AF model parameters

nsteps=500 # Max number of iterations
nsuccess=50 # Number of accumulated successful samplings for kinase (50)
nclust=256
learning_rate=0.0001 

python ../afexplore/afexplore_optim.py \
       --rawfeat_dir $feat_dir \
       --output_dir $seq_dir \
       --afparam_dir $afparam_dir \
       --nclust $nclust \
       --nsteps $nsteps \
       --target_state $target_state \
       --num_success $nsuccess \
       --protein_type $protein_type \
       --learning_rate $learning_rate 
