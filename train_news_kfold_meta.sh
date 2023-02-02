#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1:cl_adan=True
#PBS -N ctdc-kfold

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
DATADIR=/storage/plzen1/home/barticka/dp
PYTHON_SCRIPT=train_news_kfold.py

export HOME=/storage/brno2/home/barticka

singularity shell --nv $CONTAINER
module add conda-modules
conda activate torch

cd $DATADIR

DIRNAME="$MODEL"-ctdc-kfold

python $PYTHON_SCRIPT --model_name $MODEL \
--output_dir $DIRNAME \
--lr 1e-5 \
--batch_size 4 \
--epochs 5 \
