#!/bin/bash
declare -a MODELs=("UWB-AIR/Czert-B-base-cased" "Seznam/small-e-czech" "mMiniLMv2-L6-H384" "mMiniLMv2-L12-H384")
declare -a COUNT=(1)

for MODEL in ${MODELs[@]}; do
        for C in ${COUNT[@]}; do
                qsub -v MODEL=$MODEL train_news_meta.sh
        done
done
