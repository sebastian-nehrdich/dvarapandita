import sentencepiece as spm
import ctranslate2
from utils.intern_transliteration import unicode_to_internal_transliteration
from utils.constants import *
import re
import os
import pandas as pd
from utils.stemming_tib import tib_process_orig_line, prepare_tib
from utils.stemming_skt import prepare_skt


def prepare_english(string):
    # split into sentences at punctuation
    sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", string)
    return sentences


def chunk_line(line, maxlen, lang):
    gap = ""
    if lang == "tib":
        gap = " "
    line_chunks = []
    chunk = []
    tokens = line
    if lang == "tib":
        tokens = line.split(' ')
    last_index = len(tokens) - 1
    for index, token in enumerate(tokens):
        chunk.append(token)
        if index == last_index or len(chunk) > maxlen:
            line_chunks.append(gap.join(chunk))
            chunk = []
    if len(chunk) > 0:
        line_chunks.append(chunk)
    return line_chunks


def create_fname(text_path):
    filename = os.path.basename(text_path)
    filename = filename.replace(".txt","")
    filename = filename.replace(".TXT","")
    filename = filename.replace(":","-") 
    return filename 


def crop_lines(filepath, lang):
    """makes the longer lines shorter and merges shorter lines"""
    lines = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            chunks = chunk_line(line, MAX_SEQ_LENGTH[lang], lang)
            lines.extend(chunks)
    if lang == "chn":
        return lines
    else:
        # second iteration for tib and skt only: merge lines that are shorter than maxlen / 2 with the next line
        prefix = ""
        cleaned_lines = []
        for line in lines:
            current_length = len(prefix + line)
            if lang == "tib":
                current_length = len(str(prefix + " " + line).split(" "))
            if current_length < MAX_SEQ_LENGTH[lang] / 10:
                prefix += " " + line
            elif current_length < MAX_SEQ_LENGTH[lang] and re.search("dang[^a-zA-Z]*$", line.lower()):
                prefix += " " + line
            else:
                cleaned_lines.append(prefix + " " + line)
                prefix = ""
        return cleaned_lines

    return lines


def verify_orig_line(orig_line, filename):
    # Print at least a warning when encountering very long lines
    if len(orig_line) > 1000:
        print("Line too long: " + orig_line)
        print("WARNING: Very long line in file: " + filename)


def text2lists(filename, lines, lang):
    line_count = 0
    orig_lines = []
    cleaned_lines = []
    filenames = []
    line_numbers = []
    current_folio = ""
    prefix = ""
    for orig_line in lines:
        orig_line = prefix + orig_line
        if not re.search(r"[a-zA-Z]", orig_line):  # [1] lines without text (e.g. only numbers) are skipped -- should be extended with diacritica!!!!!! make a separate func
            prefix += (orig_line.strip() + " ")  # the exact form of the orig line should be saved!!!
        else:
            prefix = ""
            # Prepeare original (only tib) and create cleaned line
            #################################################################################
            if lang == "tib":
                orig_line, current_folio, line_number, line_count = tib_process_orig_line(filename, orig_line, current_folio, line_count)
                orig_line = orig_line.strip() # BAD: now "original" tibetan line is modified two times: by tib_orig_line_preparation and by this stripping
                cleaned_line = prepare_tib(orig_line)
            else:
                line_number = str(line_count)
                if lang == "skt":
                    cleaned_line = prepare_skt(orig_line)
            ################################################################################

            if not re.search(r"[a-zA-Z]", cleaned_line):  # [2] the cleaned line is check if empty
                prefix += (orig_line.strip() + " ")  # the exact form of the orig line should be saved!!!
            else:
                verify_orig_line(orig_line, filename)
                orig_lines.append(orig_line)
                cleaned_lines.append(cleaned_line)
                filenames.append(filename)
                line_numbers.append(line_number)
            line_count += 1

    return [filenames, line_numbers, orig_lines, cleaned_lines]
