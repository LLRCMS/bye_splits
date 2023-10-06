#!/usr/bin/env bash

cd /home/llr/cms/ehle/NewRepos/bye_splits/bye_splits/scripts/cluster_size/condor/

# Coefficients (radii) stored in .txt file, run cluster step on each radius
coef_file=$1
particles=$2
pileup=$3
while read -r line; do
    python run_cluster.py --coef "$line" --particles "$particles" --pileup "$pileup"
done <$coef_file