import os
import pandas as pd
import re
import numpy as np
import multiprocessing

from utils.constants import *
from utils.stemming import *
from utils.stemming_skt import skt_stemming
from utils.stem_chinese import *

from utils.stem_chinese import stem_chinese_file
from utils.general import test_if_should_load

from pathlib import Path


def testing_write(fun):
    print(">>> {} >>>".format(os.environ.get("DP_TESTMODE")))

    def wrapper(df, path):
        path = Path(path).absolute()
        (path.parent.parent / "testing-stemmed").mkdir(exist_ok=True)
        fun(df, str(path).replace("original-raw", "testing-stemmed"))

    return wrapper


@testing_write
def write_df(df, path):
    df["segmentnr"] = df["filename"] + ":" + df["line_number"]
    # write tsv files in chunks
    print(df)
    for num, chunk in df.groupby(np.arange(len(df)) // TEXT_CHUNKSIZE):
        print("NOW WRITING", path + "${}.tsv".format(num))
        chunk.to_csv(
            path + "${}.tsv".format(num),
            sep="\t",
            index=False,
            columns=["segmentnr", "original", "stemmed"],
        )


def stem_file(data):
    path, lang = data
    print("NOW PROCESSING", path)
    cfile = open(path, "r")
    path_short = os.path.splitext(path)[0]
    lines = crop_lines(path, lang)
    filename = create_fname(path)
    filenames, line_numbers, lines, cleaned_lines = text2lists(filename, lines, lang)
    text_df = pd.DataFrame(
        {
            "filename": filenames,
            "line_number": line_numbers,
            "original": lines,
            "stemmed": cleaned_lines,
        }
    )
    if lang == "skt":
        text_df = skt_stemming(text_df)  # padaccheda
    write_df(text_df, path_short)


def preprocess_translated_file(path):
    print("NOW PROCESSING", path)
    current_df = pd.read_csv(
        path,
        sep="\t",
        names=["original", "stemmed"],
        on_bad_lines="skip",
        engine="python",
    ).astype(str)
    # split stemmed string into list at punctuation marks, preserve punctuation
    current_df["stemmed"] = current_df["stemmed"].apply(prepare_english)
    current_df = current_df.explode("stemmed")
    current_df = current_df.dropna()
    path_short = os.path.splitext(path)[0]
    filename = os.path.basename(path_short)
    current_df["filename"] = filename
    line_numbers = []
    for i in range(len(current_df)):
        line_numbers.append(str(i))
    current_df["line_number"] = line_numbers
    write_df(current_df, path_short)


def run_stemmer(path, lang, num_of_threads):
    print("STARTING STEM PROCESS")
    list_of_paths = []
    for cfile in os.listdir(path):
        if not test_if_should_load(cfile):
            continue
        filename = os.fsdecode(cfile)
        # make sure we only read txt-files for skt and tib
        if lang == "skt" or lang == "tib":
            if ".txt" in filename and not os.path.isfile(
                path + filename.replace(".txt", "$0.tsv")
            ):
                list_of_paths.append([path + filename, lang])
        if lang == "chn":
            if ".json.gz" in filename:
                list_of_paths.append([path+filename,lang])
        if lang == "eng":
            if ".tsv" in filename and not "$" in filename:
                list_of_paths.append(path + filename)

    pool = multiprocessing.Pool(processes=num_of_threads)
    if lang == "skt" or lang == "tib":
        quote_results = pool.map(stem_file, list_of_paths)
    if lang == "chn":
        quote_results = pool.map(stem_chinese_file, list_of_paths)
    if lang == "eng":
        quote_results = pool.map(preprocess_translated_file, list_of_paths)
    pool.close()
