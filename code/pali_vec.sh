#!/bin/bash    
#PBS -l select=1:ncpus=160

source /homes/nehrdich/pali-dp/venv/bin/activate
echo path ${PATH}
INPUT_DIR_PATH="/tier2/ucb/nehrdich/pli/stemmed/" \
OUTPUT_ROOT="/tier2/ucb/nehrdich/" \
N_BUCKETS=10 N_PROC=160 \
python /homes/nehrdich/pali-dp/code/pali_vectorize_all.py
