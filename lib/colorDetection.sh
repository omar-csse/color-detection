#!/bin/bash -l

#PBS -N color_detection
#PBS -l ncpus=8
#PBS -l ngpus=4
#PBS -l gputype=P100
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
cd $PBS_O_WORKDIR    

module load tensorflow/1.5.0-gpu-p100-foss-2018a-python-3.6.4
pip3 install --upgrade h5py --user
python colorDetection.py