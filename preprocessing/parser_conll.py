"""parser_conll.py: Parser corpus following CONLL format"""
__author__ = "Luis Espinosa Anke, Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"


import os
import multiprocessing as mp
import spacy
from nltk.tokenize import sent_tokenize
from argparse import ArgumentParser


SENTENCES_FILE = 'sentences.txt'
nlp = spacy.load('en_core_web_sm')

def split_sentences(corpus_path):
    print('Working with corpus_path: ',corpus_path)
    with open(corpus_path) as infile, open(SENTENCES_FILE, 'w') as outFile:
        for line in infile:
            outFile.write(sent_tokenize(line)[0] +"\n")



def parser_corpus_conll(sentences_path):
    index = str(sentences_path).index("split_")
    out_file_path = os.path.join(output, str(sentences_path)[index:])

    try:
        with open(sentences_path) as infile, open(out_file_path, 'w') as outFile:
            for line in infile:
                doc = nlp(line)
                for token in doc:
                    out = (token.text + " " + str(token.i) + " " + token.pos_ +  " " +  token.dep_ + " "+ str(token.head.i) + "\n")
                    outFile.write(out)
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
        outf = open(os.path.join(splitted, 'split_' + str(splitcount)) + '.txt', 'w')
        for line in f:
            linecount += 1
            outf.write(line)
            if linecount % sentence_split == 0:
                outf.close()
                splitcount+=1
                outf =open(os.path.join(splitted, 'split_' + str(splitcount)) + '.txt', 'w')



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--corpus_file', help='Corpus file', required=True)
    parser.add_argument('-t', '--threads', help='Threads to use', required=False)
    parser.add_argument('-o', '--output', help='Corpus file', required=True)
    parser.add_argument('-s', '--splitted', help='Corpus file', required=True)

    args = parser.parse_args()
    corpus = args.corpus_file
    output = args.output
    splitted = args.splitted


    if not args.threads:
        workers = mp.cpu_count()
    else:
        workers = int(args.threads)

    if not os.path.exists(splitted):
        os.makedirs(splitted)

    if not os.path.exists(output):
        os.makedirs(output)

    split_sentences(corpus)
    workers = mp.cpu_count()
    sents_per_split = split_workload(workers)
    generate_splitted_files(sents_per_split)


    p = mp.Pool(processes=workers)
    splits=[os.path.join(splitted,inf) for inf in os.listdir(splitted) if
              inf.startswith('split') and inf.endswith('.txt')]

    p.map(parser_corpus_conll, splits)
    p.close()

