from pathlib import Path
import glob
import os 
import numpy as np
import re
import faiss
import pandas as pd
from utils.constants import *
from utils.general import test_if_should_load
from merge_results import process_matches
import multiprocessing
import multiprocessing.pool


class CalculateResults:
    def __init__(self, bucket_path, lang, index_method="cpu", cindex=None, alignment_method="local"):
        self.main_path = re.sub("folder.*","",bucket_path)
        self.wordlist = pd.read_pickle(bucket_path + "wordlist.p")
        self.segments = self.wordlist['segmentnr'].tolist()
        self.stems = self.wordlist['stemmed'].tolist()
        self.sentences = self.wordlist['original'].tolist()
        self.wordlist = self.wordlist.reset_index()
        self.wordlist_len = len(self.wordlist)
        self.lang = lang
        self.bucket_path = bucket_path
        self.alignment_method = alignment_method
        self.index_method = index_method

        global index
        if index_method == "cpu":
            index = faiss.read_index(bucket_path + "vectors.idx")
        else:
            index = cindex   


    def clean_results_by_threshold(self, scores, positions):
        current_positions = []
        current_scores = []
        list_of_accepted_numbers = []
        threshold = THRESHOLD[self.lang]
        bigger_is_better = BIGGER_IS_BETTER[self.lang]
        if bigger_is_better:
            condition = lambda score: score > threshold
        else:
            condition = lambda score: score < threshold
            
        for current_position,current_score in zip(positions,scores):
            if condition(current_score):
                current_positions.append(current_position)
                current_scores.append(current_score)
        return [current_positions,current_scores]

    ### debug this oct 12, we have a problem with the results being shifted +1 which is bad 
    def get_word_data(self, query_position, positions, scores):
        def get_word_data_from_position(position, score, is_end=False):
            segment = self.segments[position]
            sentence = self.sentences[position]
            
            # Adjust stem data based on language
            stems = self.stems[position]
            if self.lang != "eng":                
                stems = " ".join(str(item) if isinstance(item, float) else item for item in self.stems[position:position + WINDOWSIZE[self.lang]])
            
            data = [segment, sentence, stems, position, score]
            
            # If it's the end position and data is new, append to the list
            if is_end and data not in all_word_data:
                all_word_data.append(data)

        all_word_data = []

        for position, score in zip(positions, scores):
            if position >= 0:  # faiss returns -1 when not enough results are found
                get_word_data_from_position(position, score)
                
                end_position = position + WINDOWSIZE[self.lang]
                if end_position < self.wordlist_len:
                    get_word_data_from_position(end_position, score, is_end=True)        
        return all_word_data


    def is_result_in_parent_dir(self, filepath, query_path):
        return os.path.isfile(filepath.replace(".p",".json.gz").replace(query_path,self.bucket_path))

    def create_querypaths(self, query_path):
        filelist =  glob.glob(query_path + '/**/*.p', recursive=True)
        filepaths = []
        for current_file in filelist:
            filepath = current_file
            if test_if_should_load(filepath):
                if not self.is_result_in_parent_dir(filepath, query_path) and not "wordlist" in filepath:
                    filepaths.append(filepath)
                else:
                    print(f"Calculator: Skipping {filepath}")
                    
        return filepaths

    def calc_results_folder(self, query_path):
        query_files = self.create_querypaths(query_path)
        # GPU faiss is not thread safe, so we have to do it sequentially
        if self.index_method == "gpu":
            for query_file in query_files:  
                self.calc_results_file(query_file)     
        # cpu faiss is thread safe, so we can use multiprocessing
        else:   
            def handle_result(result):
                # Do something with the result
                pass

            # Function to handle error (optional)
            def handle_error(e):
                print(f"Error: {e}")
            results = []
            #pool = multiprocessing.pool.ThreadPool(processes=40)
            pool = multiprocessing.Pool(processes=16)
            results = pool.map(self.calc_results_file, query_files)
            # Iterate over each file and apply `self.calc_results_file` asynchronously
            # it rarely happens that local alignment times out due to hard to trace bugs in the Biopython aligner library; for that reason, we need to set a timeout for each worker process to make sure that we are not stalling indefinitely
            
            #for query_file in query_files:
            #    result = pool.apply_async(self.calc_results_file, args=(query_file,),
            #                            callback=handle_result, error_callback=handle_error)
            #    results.append(result)    

            # Close the pool and no longer accept new tasks
            
            #pool.join()
            #pool.close()

            # Wait for each task to complete with a timeout
            #for result in results:
            #    try:
                    # Wait for each result with a timeout of 1800 seconds
            #        result.get(timeout=1800)
            #    except multiprocessing.TimeoutError:
            #        print("A task exceeded the 1800 seconds timeout.")

            # Join the pool to wait for worker processes to exit
            pool.close()
            pool.join()
            
        
    def calc_results_file(self, query_file_path):
        # Initialize result DataFrame and retrieve query data
        print(f"NOW PROCESSING {query_file_path}")
        basename = Path(query_file_path).stem
        # try to read the file, if it is broken, skip it
        try:
            query_df = pd.read_pickle(query_file_path)
        except:
            print(f"ERROR READING {query_file_path}")
            return        
        
        query_vectors = self.prepare_query_vectors(query_df)
        
        if query_vectors is not None:
            query_results = index.search(query_vectors, QUERY_DEPTH)
            
            # Extract results
            results = self.extract_results(query_df, query_results)
            
            # Convert results to DataFrame and save to JSON
            result_df = pd.DataFrame(results)
            #self.save_results_to_json(result_df, basename)

            # Further processing
            process_matches(query_df, result_df, self.bucket_path + basename, self.lang, alignment_method=self.alignment_method)

    def prepare_query_vectors(self, query_df):
        query_vectors = np.array(query_df['sumvectors'].tolist(), dtype="float32")
        if len(query_vectors) > 0:
            faiss.normalize_L2(query_vectors)
            return query_vectors

    def extract_results(self, query_df, query_results):
        results = {
            "query_position": [],
            "query_segmentnr": [],
            "query_stems": [],
            "query_sentence": [],
            "match_segment": [],
            "match_sentence": [],
            "match_stems": [],
            "match_position": [],
            "match_score": []
        }
        
        for query_position, (scores, positions) in enumerate(zip(query_results[0], query_results[1])):            
            cleaned_positions, cleaned_scores = self.clean_results_by_threshold(scores, positions)        
            word_data = self.get_word_data(query_position, cleaned_positions, cleaned_scores)
            current_query_data = self.get_current_query_data(query_df, query_position)
            for entry in word_data:
                results["query_position"].append(query_position)
                results["query_segmentnr"].append(current_query_data["segmentnrs"])
                results["query_stems"].append(current_query_data["stems"])
                results["query_sentence"].append(current_query_data["sentence"])
                results["match_segment"].append(entry[0])
                results["match_sentence"].append(entry[1])
                results["match_stems"].append(entry[2])
                results["match_position"].append(entry[3])
                results["match_score"].append(entry[4])
                
        return results

    def get_current_query_data(self, query_df, query_position):
        total_query_stems = query_df['stemmed']
        total_query_segmentnrs = query_df['segmentnr']
        total_query_sentences = query_df['original']
        query_position = min(query_position, len(total_query_stems))
        end_position = min(query_position + WINDOWSIZE[self.lang], len(total_query_stems))
        stems = " ".join(str(item) for item in total_query_stems[query_position:end_position])
        segmentnrs = list(dict.fromkeys(total_query_segmentnrs[query_position:end_position]))
        sentence = " ".join(total_query_sentences[query_position:end_position])
        
        if self.lang == "eng":
            stems = total_query_stems[query_position]
            segmentnrs = [total_query_segmentnrs[query_position]]
            
        return {"stems": stems, "segmentnrs": segmentnrs, "sentence": sentence}

    def save_results_to_json(self, result_df, basename):
        result_df.to_json(
            self.bucket_path + basename + "-before-alignment.json.gz", 
            orient="records", force_ascii=False, 
            compression="gzip", indent=2
        )


    def run(self):
        for directory in os.listdir(self.main_path):            
            print(f"Calculator: directory: {directory}")
            if Path(self.main_path+directory).is_dir():
                self.calc_results_folder(self.main_path+directory)
            else:
                print(f"Calculator: directory {directory} is not folder!")                                        