#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH -w gnode055
#SBATCH --output=alexnet_logs.txt

cd /scratch/pritk/
source venv/bin/activate
python alexnet.py