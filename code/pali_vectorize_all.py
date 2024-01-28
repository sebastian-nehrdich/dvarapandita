from code.pali_vectorizer import Vectorizer
from code.pali_file_mngr import FileMngr

import multiprocessing
import os

input_dir_path = os.environ["INPUT_DIR_PATH"]
output_root_path = os.environ["OUTPUT_ROOT"]
n_buckets = int(os.environ["N_BUCKETS"])
n_proc = int(os.environ["N_PROC"])

fm = FileMngr(
    n_buckets=n_buckets,
    output_root_path=output_root_path,
    ref_path="/homes/nehrdich/pali-dp/code/ref/"
)
fm.stemmed_path['pali'] = input_dir_path

lang = "pli"

def vectorize_text(file_path):
    Vectorizer(fm, lang).process_text(file_path)

list_of_paths = fm.get_stemmed_files(lang)
pool = multiprocessing.Pool(processes=n_proc)
pool.map(vectorize_text, list_of_paths)
pool.close()