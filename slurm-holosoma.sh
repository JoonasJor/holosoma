#!/bin/bash
#SBATCH --account=project_2020755
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1

cd /scratch/project_2020755/

module load python-data/3.10-17.04

source holosoma-venv/bin/activate

srun python /projappl/project_2020755/holosoma/src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:mjwarp logger:wandb --training.seed 1 --logger.video.enabled False
