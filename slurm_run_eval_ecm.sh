#!/bin/bash


#SBATCH --output ./scratch/sbatch/job%j-$1.out # Output file name
#SBATCH --error ./scratch/sbatch/job%j-$1.err # Error log file name
## Below is for requesting the resource you want
#SBATCH --nodes=1 # Number of nodes required
## SBATCH --exclusive  --gres=gpu:1 # Number of GPUs required
#SBATCH --gres=gpu:1
## SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=4 # Number of CPUs per task
#SBATCH --mem=40gb
#SBATCH --time 4:00:00
#SBATCH --partition=gpu 
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket
#SBATCH --qos=batch-short

## conda activate tsl
# python run_eval.py --data ETTh1 --model TimeMixer
python run_eval.py --data $1 --model $2