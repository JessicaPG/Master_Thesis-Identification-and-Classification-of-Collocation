"""parser.py: Parser corpus"""
__author__ = "Luis Espinosa Anke, Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"

import os
import multiprocessing as mp
import re
import spacy
from nltk.tokenize import sent_tokenize
from argparse import ArgumentParser

CORPUS_ORIG = ""
SPLITTED_DIR = "/Users/jessie/PycharmProjects/Thesis/splitted"
OUT_DIR = "/Users/jessie/PycharmProjects/Thesis/out"
SENTENCES_FILE = 'sentences.txt'



nlp = spacy.load('en_core_web_sm')


def split_sentences(corpus_path):
    print('Working with corpus_path: ',corpus_path)
    with open(corpus_path) as infile, open(SENTENCES_FILE, 'w') as outFile:
        linecount=0
        for line in infile:
            #print(sent_tokenize(line))
            outFile.write(sent_tokenize(line)[0] +"\n")
            linecount+=1


def parser(sentences_path):
    pre_filter = r'^\[["\']{1}'
    post_filter = r'["\']{1}[\]]$'

    index = str(sentences_path).index("split_")
    out_file_path = os.path.join(OUT_DIR, str(sentences_path)[index:])

    try:
        with open(sentences_path) as infile, open(out_file_path, 'w') as outFile:
            for line in infile:
                clean_line = re.sub(pre_filter, "", line, count=1)
                clean_line = re.sub(post_filter, "", clean_line, count=1)
                doc = nlp(clean_line)
                out = ['_'.join((t.lemma_, t.pos_, t.dep_)) for t in doc]
                outFile.write(' '.join(out))

    except BaseException as e:
        print("Error occurred is:  %s" % str(e))


def split_workload(workers):
    print('Counting lines in original corpus')
    linecount=0
    with open(SENTENCES_FILE) as f:
        for line in f:
            linecount+=1

    sents_per_split=round(linecount/workers)

    print('Source corpus has ',linecount,' lines')
    print('Splitting original corpus in ',workers,' files of ~',sents_per_split,' lines')
    return sents_per_split


def generate_splitted_files(sentence_split):
    linecount=0
    splitcount=0
    with open(SENTENCES_FILE,'r') as f:
        outf = open(os.path.join(SPLITTED_DIR, 'split_' + str(splitcount)) + '.txt', 'w')
        for line in f:
            linecount += 1
            outf.write(line)
            if linecount % sentence_split == 0:
                outf.close()
                splitcount+=1
                outf =open(os.path.join(SPLITTED_DIR, 'split_' + str(splitcount)) + '.txt', 'w')


#
# if __name__ == '__main__':
#
#     parser = ArgumentParser()
#     parser.add_argument('-c', '--corpus_file', help='Corpus file', required=True)
#     parser.add_argument('-t', '--threads', help='Threads to use', required=False)
#
#     args = parser.parse_args()
#
#     if not args.threads:
#         workers = mp.cpu_count()
#     else:
#         workers = int(args.threads)



if not os.path.exists(SPLITTED_DIR):
    os.makedirs(SPLITTED_DIR)

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

split_sentences(CORPUS_ORIG)
workers = mp.cpu_count()
sents_per_split = split_workload(workers)
generate_splitted_files(sents_per_split)


p = mp.Pool(processes=workers)
splits=[os.path.join(SPLITTED_DIR,inf) for inf in os.listdir(SPLITTED_DIR) if
          inf.startswith('split') and inf.endswith('.txt')]

result = p.map(parser, splits)
p.close()