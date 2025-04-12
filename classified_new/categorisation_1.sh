#!/bin/bash
#SBATCH --job-name=LLaMA3_prompt
#SBATCH --account=stats_dept1
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --partition=spgpu
#SBATCH --time=4:00:00
#SBATCH --output=llama3_%A_%a.out
#SBATCH --error=llama3_%A_%a.err
#SBATCH --array=2-4

# Define range based on array ID
START=$(( SLURM_ARRAY_TASK_ID * 1000 ))
END=$(( START + 1000 ))

echo "Running for indices $START to $END"

# python3 categorisation.py --start_index $START --end_index $END

python3 categorisation.py --start_index 5000 --end_index 5273