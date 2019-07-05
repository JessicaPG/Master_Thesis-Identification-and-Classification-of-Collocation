# -*- coding: utf-8 -*-
"""filter.py: Filter pair of words (candidate to collocatin) by syntactic pattern using Regular Expressions"""
__author__ = "Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"

import os
from argparse import ArgumentParser
import re
import multiprocessing as mp


#Syntactic pattern definition
pattern1 = re.compile(r"([^\s]+)VERB([^\s]+)(\s)([^\s]+)dobj") #Pattern --> VERB + dobj
pattern2 = re.compile(r"([^\s]+)_amod(\s)([^\s]+)NOUN") #Pattern --> amod + NOUN
pattern3 = re.compile(r"([^\s]+)VERB(\s)([^\s]+)(\s)([^\s]+)dobj") #Pattern --> VERB + sth + dobj
pattern4 = re.compile(r"([^\s]+)_amod(\s)([^\s]+)(\s)([^\s]+)NOUN") #Pattern --> amod + sth + NOUN

list_patterns = [pattern1,pattern2,pattern3,pattern4]


def match_pattern(sentence, pairs_words):
    """""
    input: sentence, dictionary 
    Find all the matches for the patterns in list_patterns for the input sentence and save the pair of words in dict

    """""
    sentence = sentence.replace("_SPACE_", "")
    for pattern in list_patterns:
        result = re.findall(pattern,sentence)

        for match in result:
            base = match[0].split("_")[0]

            #For patterns 3 and 4 make a two words col
            if pattern == pattern3 or pattern == pattern4:
                col = match[2].split("_")[0] + " " + match[len(match)-1].split("_")[0]
            else:
                col = match[len(match) - 1].split("_")[0]

            key = base + "_" + col
            pairs_words[key] = (base, col)

    return pairs_words

def preprocess_corpus(infile, data_type='txt'):
    """""
    Main filter function
    """""
    pairs_words = {}
    index = str(infile).index("split_")
    out_file_path = os.path.join(output_filt, str(infile)[index:])
    try:
        with open(infile) as inf,open(out_file_path, 'w') as outFile:
            for line in inf:
                match_pattern(line, pairs_words)
            for key in pairs_words:
                outFile.write(pairs_words[key][0] + " " + pairs_words[key][1] + "\n")
    except BaseException as e:
        print("Error occurred is:  %s" % str(e))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-pc', '--parsed_corpus', help='Corpus dir', required=True)
    parser.add_argument('-o', '--output_filtered', help='output dir', required=True)

    args = parser.parse_args()
    corpus_parsed_dir = args.parsed_corpus
    output_filt = args.output_filtered

    #
    # if not args.threads:
    #      workers = mp.cpu_count()
    # else:
    #      workers = int(args.threads)

    if not os.path.exists(output_filt):
        os.makedirs(output_filt)

#Concurrence process
workers = mp.cpu_count()

p = mp.Pool(processes=workers)
splits=[os.path.join(corpus_parsed_dir,inf) for inf in os.listdir(corpus_parsed_dir) if
           inf.startswith('split') and inf.endswith('.txt')]

p.map(preprocess_corpus, splits)
p.close()