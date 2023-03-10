#! /bin/bash

user=$1

python bye_splits/scripts/cluster_size.py --reg All --sel all --user "$1" --cluster_studies True