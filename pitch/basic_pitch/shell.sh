#!/bin/bash
#SBATCH --job-name=basic
#SBATCH -n 2
#SBATCH --gres=gpu:quadro:1
# #SBATCH --mem 10000
#SBATCH -p high                     # Partition to submit to
#SBATCH -o %x-%j.out # File to which STDOUT will be written
#SBATCH -e %x-%j.err # File to which STDERR will be written

module --ignore-cache load "libsndfile"
module --ignore-cache load "FFmpeg/4.3.2-GCCcore-10.2.0"
module --ignore-cache load "Anaconda3"
module --ignore-cache load "TensorFlow/2.7.1-foss-2020b-CUDA-11.4.3"
module --ignore-cache load "CUDA"
module --ignore-cache load "cuDNN"

pip install mir_eval
pip install basic-pitch



python extract_basic_pitch.py

