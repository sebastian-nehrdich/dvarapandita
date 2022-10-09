import pandas as pd 
import numpy as np 
import re
import faiss
import os 
from utils.constants import *
from utils.indexing import calculate_results


def create_index(bucket_path):
    global total_vectors
    all_files = pd.DataFrame()
    for file in os.listdir(bucket_path):
        if ".p" in file and not "wordlist" in file:            
            file_path = bucket_path + file
            print("NOW LOADING",file_path)
            file_df = pd.read_pickle(file_path)
            all_files = pd.concat([all_files,file_df])
    total_vectors = np.array(all_files['sumvectors'].tolist(),dtype="float32")

    # wordlist is used in conjunction with the index so that the individual tokens on the index can be identified; we can drop all vector data to save discspace. 
    wordlist = all_files.drop(["vectors","sumvectors",'weights'],axis=1)
    wordlist.to_pickle(bucket_path + "wordlist.p")

    # build the index
    dim = total_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.verbose = True
    faiss.normalize_L2(total_vectors)
    index.add(total_vectors)
    faiss.write_index(index, bucket_path + "vectors.idx")

        
    


# provisional for debug; control this via Makefile later    
#create_index("../tibetan-work/folder1/")

def run_calculation(path, lang):
    calculator = calculate_results(path, lang)
    calculator.run()

