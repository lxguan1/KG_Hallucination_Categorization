#!/bin/bash
#SBATCH --job-name=KG_creation
#SBATCH --account=cse598s012w25_class
#SBATCH --mem=7gb   
#SBATCH --gpus=1
#SBATCH --partition=spgpu
#SBATCH --time=5-12:00:00
#SBATCH --array=100,200,300,400

echo 'This job runs knowledge graph creation'

python3 temp_proj_script.py --start_document $SLURM_ARRAY_JOB_ID --end_document $(($SLURM_ARRAY_JOB_ID + 20))





