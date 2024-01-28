from Bio import Align
import re
from utils.constants import PUNC, TIBETAN_STEMFILE
from pathlib import Path
import os

def create_aligner(lang):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    # todo: the following parameters need to be adjusted based on the individual languages, as the parameters are unlikely to fit well for all cases.
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -5
    return aligner

def create_replaces_dictionary(path):
    replaces_dictionary = {}
    utils_dir = os.path.dirname(os.path.realpath(__file__)) # find the file's dir
    path = Path(utils_dir).parent / path
    r = open(path, "r")
    for line in r:
        headword = line.split('\t')[0]
        entry = line.split('\t')[2]
        replaces_dictionary[headword] = entry.strip()
    return replaces_dictionary

replaces_dictionary = create_replaces_dictionary(TIBETAN_STEMFILE)

def multireplace(string):
    if string in replaces_dictionary:
        string = replaces_dictionary[string]
    return string


def crude_stemmer(tokens):
    tokens = [multireplace(x) for x in tokens]
    result_tokens = []
    for token in tokens:
        token = re.sub(r"([a-z])\'.*", r"\1", token)
        if "/" in tokens:
            token = token + str(random.randint(1, 100))
        result_tokens.append(token)
    return result_tokens


def get_aligned_offsets_efficient(inquiry_text, target_text, threshold, lang, aligner):
    # efficient version of get_aligned_offsets where we only look at the first 100 tokens in both directions, avoiding the algorithm to get stuck on very long matches
    if len(inquiry_text) < threshold or len(target_text) < threshold:
        return get_aligned_offsets(inquiry_text, target_text, lang, aligner)
    else:
        inquiry_text_beg, w, target_text_beg, v, score_beg = get_aligned_offsets(inquiry_text[:threshold],
                                                                         target_text[:threshold], lang, aligner)
        x, inquiry_text_end, y, target_text_end, score_end = get_aligned_offsets(inquiry_text[-threshold:],
                                                                         target_text[-threshold:], lang, aligner)
        score = (score_beg + score_end) / 2
        inquiry_text_end = inquiry_text_end + len(inquiry_text) - threshold
        target_text_end = target_text_end + len(target_text) - threshold
        return inquiry_text_beg, inquiry_text_end, target_text_beg, target_text_end, score


def get_aligned_offsets(stringa, stringb, lang, aligner):
    if len(stringa) <=3 or len(stringb) <=3:
        print("Too short strings for alignment: ", stringa, stringb)
        return 0, 0, 0, 0, 0
    stringa_tokens_before = stringa
    stringb_tokens_before = stringb
    if lang == "tib":
        stringa_tokens_before = stringa.split()
        stringb_tokens_before = stringb.split()

    c = 0
    stringa_lengths = []
    stringa_tokens_after = []
    for token in stringa_tokens_before:
        if not "/" in token and not "@" in token and token not in PUNC and not re.search('[0-9]', token):
            stringa_lengths.append(c)
            stringa_tokens_after.append(token)
        if lang == "tib":
            c += len(token) + 1
        else:
            c += len(token)
    c = 0
    stringb_lengths = []
    stringb_tokens_after = []
    for token in stringb_tokens_before:
        if not "/" in token and not "@" in token and token not in PUNC and not re.search('[0-9]', token):
            stringb_lengths.append(c)
            stringb_tokens_after.append(token)
        if lang == "tib":
            c += len(token) + 1
        else:
            c += len(token)

    stringa_tokens = crude_stemmer(stringa_tokens_after)
    stringb_tokens = crude_stemmer(stringb_tokens_after)
    alphabet = list(set(stringa_tokens + stringb_tokens))
    aligner.alphabet = alphabet

    stringa_lemmatized = " ".join(stringa_tokens)
    stringb_lemmatized = " ".join(stringb_tokens)
    try:
        alignments = aligner.align(stringa_tokens, stringb_tokens)
        if alignments:
            stringa_beg = alignments[0].aligned[0][0][0]
            stringa_end = alignments[0].aligned[0][-1][1]
            stringb_beg = alignments[0].aligned[1][0][0]
            stringb_end = alignments[0].aligned[1][-1][1]

            stringb_beg = stringb_lengths[stringb_beg]
            if stringb_end < len(stringb_lengths):
                stringb_end = stringb_lengths[stringb_end]
            else:
                stringb_end = len(stringb)

            stringa_beg = stringa_lengths[stringa_beg]
            if stringa_end < len(stringa_lengths):
                stringa_end = stringa_lengths[stringa_end]
            else:
                stringa_end = len(stringa)

            return stringa_beg, stringa_end, stringb_beg, stringb_end, alignments[0].score
        else:
            return 0, 0, 0, 0, 0
    except Exception as e:
        print("Error during alignment: ", e)
        return 0, 0, 0, 0, 0
