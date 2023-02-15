#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1:cl_adan=True
#PBS -N sst-attrs

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
DATADIR=/storage/plzen1/home/barticka/dp
PYTHON_SCRIPT=create_attributions_sst.py

export HOME=/storage/brno2/home/barticka

singularity shell --nv $CONTAINER
module add conda-modules
conda activate torch

cd $DATADIR

OUT_DIRNAME="$MODEL"-attrs

python $PYTHON_SCRIPT --output_dir $OUT_DIRNAME \
--model_path $MODEL --baselines_dir "$MODEL"/baselines-sst