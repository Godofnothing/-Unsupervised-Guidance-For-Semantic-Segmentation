#!/bin/bash

#SBATCH --job-name UnsupervisedSS
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --time 4:00:00 # hh:mm:ss, walltime
#SBATCH --mem 16G

ROOT_DIR=/trinity/home/d.kuznedelev/Datasets/Intel_Image_Classification

source /home/${USER}/.bashrc
source activate mmlab

python kNNRelabeling.py \
    --annotation_dir ${ROOT_DIR}/annotations \
    --num_clusters 40 