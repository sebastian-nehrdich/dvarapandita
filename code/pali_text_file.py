from pathlib import Path

import pandas as pd

class TextFile:



    # suffixes = {
    #     "stemmed":  "stemmed",
    #     "vectors":  "vectors"
    # }
    stemmed_extention = ".stemmed.tsv"
    vectors_extention = ".p"

    # segments_df_col_names =     ["segmentnr", "original_text", "stemmed_segment"]
    segments_df_col_names =     ["segmentnr", "original", "stemmed_segment"]
    # words_df_col_names =        ["segmentnr", "words", "weights", "vectors", "sumvectors"]
    words_df_col_names =        ["segmentnr", "stemmed", "weights", "vectors", "sumvectors"]
    on_bad_lines = "skip"

    def __init__(self, lang, input_path: Path, sep="\t"):
        self.lang: str = lang
        self.input_path: Path = input_path
        self.name: str = self.init_name()
        self.root_dir = self.input_path.parent.parent
        self.stemmed_file = self.name + self.stemmed_extention
        self.vectors_file = self.name + self.vectors_extention
        self.sep = sep
        self.original_txt = None
        self.segments_df = None # Pandas DataFrame
        self.words_df = None # Pandas DataFrame
        self.tags = [] # could be decoded from the file name

    # def verify_name(self):
    #     parts = self.path.name.split(".")
        


    def init_name(self):
        def remove_suffix(n):
            return "".join(str(self.input_path.name).split(".")[:-n])
        if self.input_path.match("*" + self.stemmed_extention):
            return remove_suffix(2)
        elif self.input_path.match("*" + self.vectors_extention):
            return remove_suffix(1)
        elif self.input_path.match("*.txt"):
            return remove_suffix(1)
        else:
            print("Bad name")
            raise Exception


    def init_segments_df(self, stemmed_path):
        self.segments_df = pd.read_csv(stemmed_path / self.stemmed_file, 
                              sep=self.sep, 
                              names= self.segments_df_col_names,
                              on_bad_lines=self.on_bad_lines).astype(str)
        return self

    def init_words_df(self):
        self.words_df = pd.read_pickle(self.vectors_path)
        return self
