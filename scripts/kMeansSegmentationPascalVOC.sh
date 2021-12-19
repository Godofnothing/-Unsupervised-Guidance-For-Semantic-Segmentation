#!/bin/bash

#SBATCH --job-name UnsupervisedSS
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --time 2:00:00 # hh:mm:ss, walltime
#SBATCH --mem 16G

DATA_DIR=/trinity/home/d.kuznedelev/Datasets/VOCdevkit/VOC2012

source /home/${USER}/.bashrc
source activate mmlab

python kMeansSegmentationPascalVOC.py \
    --data_dir ${DATA_DIR} \
    --num_clusters 6 \
    --attempts 10 \
    --annotation_dir ${DATA_DIR}/kMeans_annotations
