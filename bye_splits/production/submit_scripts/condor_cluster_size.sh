#!/usr/bin/env bash

cd ${HOME}/bye_splits/bye_splits/scripts/cluster_size/condor/

radius=()
particles=""
pileup=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --radius)
            IFS=";" read -ra radius <<< "${2:1:-1}"
            shift 2
            ;;
        --particles)
            particles="$2"
            shift 2
            ;;
        --pileup)
            pileup="$2"
            shift 2
            ;;
        *)
            echo "Unrecognized argument $1"
            exit 1;;
    esac
done

for rad in ${radius[@]}; do
    python run_cluster.py --radius "$rad" --particles "$particles" --pileup "$pileup"
done