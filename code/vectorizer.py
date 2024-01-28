import re
from pathlib import Path
import multiprocessing

import pandas as pd
import numpy as np
import fasttext

from utils.constants import *  ###################################################
from text_file import TextFile


class Vectorizer:
    """tsv(segmentId, orig-text) --> tsv(segmentId, orig-text, tokenized-text)"""

    def __init__(
        self,
        file_mngr,
        lang: str,
    ) -> None:
        self.lang: str = lang
        self.file_mngr = file_mngr

        ################## Lang specific ########################  --> LanguageMngr ???
        self.windowsize = WINDOWSIZE[lang]
        self.splitter = self.init_splitter()
        self.vector_model = self.init_vector_model()  # fasttext
        self.stopwords = self.read_stopwords()


    def process_text(self, file_path):
        text_obj = TextFile(self.lang, file_path).init_segments_df(
            self.file_mngr.stemmed_path[self.lang])
        print("NOW PROCESSING", text_obj.name)
        self.add_words_df(text_obj)
        self.add_window_vecs(text_obj)
        self.file_mngr.pickle_words_df(text_obj)
        del text_obj

    def add_words_df(self, text_obj):
        # 1. the segments are splitted into word lists
        words_df = text_obj.segments_df # text_obj.segments_df.drop(columns="original_text")
        words_df["stemmed_segment"] = words_df["stemmed_segment"].apply(self.splitter)
        # 2. the frame is "streched" by the word lists
        text_obj.words_df = words_df.explode("stemmed_segment").rename(
            columns={"stemmed_segment": "stemmed"} # "words"}
        )

    def add_window_vecs(self, text_obj):
        text_obj.words_df["weights"] = text_obj.words_df["stemmed"].apply( # "words"
            lambda word: self.get_weight(word)
        )
        text_obj.words_df["vectors"] = text_obj.words_df["stemmed"].apply( # "words"
            lambda word: self.get_vector(word)
        )
        text_obj.words_df["sumvectors"] = self.calc_win_vecs(
            text_obj.words_df["vectors"].tolist(), text_obj.words_df["weights"].tolist()
        )

    def init_vector_model(self):
        return fasttext.load_model(str(self.file_mngr.ref_path / self.lang) + ".bin")

    def read_stopwords(self):
        f = open(str(self.file_mngr.ref_path / self.lang) + "_stop.txt", "r")
        stopwords = []
        for line in f:
            m = re.search("(.*)", line)  # this is for tibetan
            # m = re.search("([^\t]+)\t(.*)", line) # this is for chinese
            if m:
                if not m[0] == "#":
                    stopwords.append(m.group(1).strip())
        return stopwords

    def get_weight(self, word):
        if word in self.stopwords:
            return 0.1
        else:
            return 1

    def get_vector(self, word):
        try:
            assert type(word) == str
            return self.vector_model.get_word_vector(word)
        except:
            print(f"{word}: type {type(word)}")

    def get_sumvector(self, vectors, weights=False):
        if weights:
            return self.vector_pool_hier_weighted(vectors, weights)
        else:
            return np.average(vectors, axis=0)

    def vector_pool_hier_weighted(self, vectors, weigths):
        pool = []
        for i in range(1, len(vectors) + 1):
            for vector in vectors[0:i]:
                pool.append(np.average(vectors[0:i], axis=0, weights=weigths[0:i]))
        return np.mean(pool, axis=0)

    def calc_win_vecs(self, vector_list, weight_list):
        sumvectors = []
        for i in range(len(vector_list)):
            k = i + self.windowsize
            sumvectors.append(self.get_sumvector(vector_list[i:k], weight_list[i:k]))
        return sumvectors

    ########## Language specific functions ##########
    def init_splitter(self):
        match self.lang:
            case "skt":
                return split_sanskrit_stem
            case other:
                return str.split

    def split_sanskrit_stem(words):
        result = []
        for stem in words.split("#"):
            stem = stem.strip().split(" ")[0]
            if len(stem) > 0:
                result.append(stem)
        return result


# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
#     def vectorize_all(self, lang):

#         list_of_paths = self.file_mngr.get_stemmed_files(lang)
#         pool = multiprocessing.Pool(processes=2) #self.file_mngr.threads)
#         pool.map(unwrap_self_vectorize_text, \
#             zip([self]*len(list_of_paths), list_of_paths))
#         pool.close()

# def unwrap_self_vectorize_text(*arg, **kwarg):
#     return Vectorizer.process_text(*arg, **kwarg)