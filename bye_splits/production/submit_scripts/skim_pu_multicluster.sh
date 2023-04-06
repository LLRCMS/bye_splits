#!/usr/bin/env bash

work_dir="/home/llr/cms/ehle/NewRepos/bye_splits/"
cd $work_dir

batch_file=$1
while read -r line; do
  ./produce.exe --inpath "$line"
done <$batch_file

exit 0