#!/bin/bash    
#PBS -l select=1:ncpus=160

source /homes/nehrdich/pali-dp/venv/bin/activate
echo path ${PATH}

cd /homes/nehrdich/pali-dp/code/
invoke calc-pali-bucket --bucket-path="/tier2/ucb/nehrdich/pli/vectors/folder0009/"