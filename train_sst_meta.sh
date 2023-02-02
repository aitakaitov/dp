#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1:cl_adan=True
#PBS -N sst

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
DATADIR=/storage/plzen1/home/barticka/dp
PYTHON_SCRIPT=run_glue.py

export HOME=/storage/brno2/home/barticka

singularity shell --nv $CONTAINER
module add conda-modules
conda activate torch

cd $DATADIR
DIRNAME="$MODEL"-sst

python $PYTHON_SCRIPT --model_name_or_path $MODEL \
--output_dir $DIRNAME \
--do_train --do_eval --max_seq_length 128 \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 --learning_rate 1e-5 --num_train_epochs 5 \
--logging_strategy epoch \
--validation_file datasets_ours/sst/dev.csv --train_file datasets_ours/sst/train.csv \
 --save_strategy epoch --evaluation_strategy epoch