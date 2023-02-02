#!/bin/bash
declare -a MODELs=("bert-base-cased" "prajjwal1/bert-medium" "prajjwal1/bert-small" "prajjwal1/bert-mini")
declare -a COUNT=(1)

for MODEL in ${MODELs[@]}; do
        for C in ${COUNT[@]}; do
                qsub -v MODEL=$MODEL train_sst_meta.sh
        done
done
