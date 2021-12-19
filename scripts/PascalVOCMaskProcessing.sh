#!/bin/bash

#SBATCH --job-name UnsupervisedSS
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --time 4:00:00 # hh:mm:ss, walltime
#SBATCH --mem 16G

DATA_DIR=/trinity/home/d.kuznedelev/Datasets/VOCdevkit/VOC2012

source /home/${USER}/.bashrc
source activate mmlab

python PascalVOCMaskProcessing.py \
    --annotation_dir ${DATA_DIR}/kMeans_annotations \
    --num_clusters 21 \
    --closure_kernel_size 5 \
    --opening_kernel_size 5