#!/bin/bash
# Do not forget to select a proper partition if the default
# one is no fit for the job! You can do that either in the sbatch
# command line or here with the other settings.
#SBATCH --job-name=sanskrit
#SBATCH --nodes=2
#SBATCH --time=1:00:00
#SBATCH --partition=std
#SBATCH --tasks-per-node=1
# Never forget that! Strange happenings ensue otherwise.
#SBATCH --export=NONE
#SBATCH --mem=64gb
#SBATCH --output=sanskrit_slurm_%j.log
set -e # Good Idea to stop operation on first error.

source /sw/batch/init.sh
source ~/.profile
# Load environment modules for your application here.

# Actual work starting here. You might need to call
# srun or mpirun depending on your type of application

for i in /work/ftsx015/tib/data/folder*; do srun  --partition=std --mem=64gb  ~/anaconda3/bin/python merge_quotes.py $i & done
sleep 10;
#for i in /work/ftsx015/tab-tib/folder*; do srun --mem=64gb --time=11:59:00 --partition=std  ~/anaconda3/bin/python merge_quotes.py $i --single & done

#srun --mem=64gb --time=11:59:00 --partition=std  ~/anaconda3/bin/python merge_quotes.py /work/ftsx015/tab-skt/folder3 --skt
