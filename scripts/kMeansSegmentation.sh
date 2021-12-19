#!/bin/bash

#SBATCH --job-name UnsupervisedSS
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --time 1:00:00 # hh:mm:ss, walltime
#SBATCH --mem 16G

ROOT_DIR=/trinity/home/d.kuznedelev/Datasets/Intel_Image_Classification

source /home/${USER}/.bashrc
source activate mmlab

python kMeansSegmentation.py \
    --train_dir ${ROOT_DIR}/seg_train/seg_train \
    --test_dir ${ROOT_DIR}/seg_test/seg_test \
    --num_clusters 5 \
    --attempts 10 \
    --annotation_dir ${ROOT_DIR}/annotations
