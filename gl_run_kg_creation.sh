#!/bin/bash
#SBATCH --job-name=KG_creation
#SBATCH --account=cse598s012w25_class
#SBATCH --mem=7gb   
#SBATCH --gpus=1
#SBATCH --partition=spgpu
#SBATCH --time=8:00:00
#SBATCH --array=390,395

echo 'This job runs knowledge graph creation'

python3 KG_Creation.py --start_document $SLURM_ARRAY_TASK_ID --end_document $(($SLURM_ARRAY_TASK_ID + 5))





