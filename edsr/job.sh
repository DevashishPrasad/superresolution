#!/bin/sh -l
# FILENAME:  job.sh

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --job-name edsr
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH --constraint=G

module load anaconda/2020.11-py38
#module load cuda/11.2.0
#module load cudnn/cuda-11.2_8.1
source activate superres
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pillow
pip install torchsummary
pip install matplotlib
pip install opencv-python
pip install pyyaml

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

python engine.py
