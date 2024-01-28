from invoke import task

from stemmer import run_stemmer 
from create_vectors import create_vectors
from calculate_index import create_index, run_calculation
from merge_for_db import merge_json_gz_files
from create_stats import extract_stats_from_files
from merge_stats import collect_stats_from_folder


@task
def stem(c, path, lang, threads=1):
    run_stemmer(path, lang,num_of_threads=threads)    

@task
def create_vectorfiles(c, tsv_path, out_path, lang, bucket_num=1, threads=1 ):
    create_vectors(tsv_path, out_path, bucket_num, lang, threads)
    
@task
def get_results_from_index(c, bucket_path, lang, index_method, alignment_method):
    run_calculation(bucket_path, lang, index_method, alignment_method)

@task
def create_new_index(c, bucket_path):
    create_index(bucket_path)

@task
def merge_results_for_db(c, input_path, output_path):
    merge_json_gz_files(input_path, output_path)

@task 
def calculate_stats(c, output_path):
    extract_stats_from_files(output_path)
    collect_stats_from_folder(output_path)
    
########################################################################

from stemmer_pali import Stemmer
import os
import sys
@task
def stem_pali(c,
              input_dir,
              model_path,
              output_dir=None,
              lang="pli"):
    stmr = Stemmer(lang=lang,
                   spm_model_path=model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                )
    stmr.process_src_dir()

# invoke vec-pali --input-dir-path="/home/wo/bn/dvarapandita/test-data/pli/stemmed" --output-dir-name="vectors" --n-buckets=1 --n-proc=6
@task
def vec_pali(c,
            input_dir_path,
            output_dir_name,
            n_buckets,
            n_proc,
             ):
    python_path = sys.executable
    print(python_path)
    os.system(f'''
                INPUT_DIR_PATH={input_dir_path} \
                OUTPUT_DIR_NAME={output_dir_name} \
                N_BUCKETS={n_buckets} \
                N_PROC={n_proc} \
                {python_path} vectorize_all_pali.py \
              ''')
    
from utils.indexing import CalculateResults
from calculate_index import create_index

# invoke calc-pali-bucket --bucket-path="/home/wo/bn/dvarapandita/test-data/pli/vectors/folder0000/"
@task
def calc_pali_bucket(c,
                    bucket_path,
              ):
        lang = "pli"
        index_method = "cpu"
        alignment_method="local"
    
        index = create_index(bucket_path, index_method)
        c = CalculateResults(bucket_path, lang, index_method, cindex=index, alignment_method=alignment_method)
        c.calc_results_folder(bucket_path)