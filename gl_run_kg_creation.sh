#!/bin/bash
#SBATCH --job-name=KG_creation
#SBATCH --account=cse598s012w25_class
#SBATCH --mem=7gb   
#SBATCH --gpus=1
#SBATCH --partition=spgpu
#SBATCH --time=5-12:00:00
#SBATCH --array=0-200:40

echo 'This job runs knowledge graph creation'

python3 KG_Creation.py --start_document $SLURM_ARRAY_JOB_ID --end_document $(($SLURM_ARRAY_JOB_ID + 20))





