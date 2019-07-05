"""train_word2vec.py: Train word embeddings model"""
__author__ = "Luis Espinosa Anke"
# import modules & set up logging
from argparse import ArgumentParser
import os
import sys
import gensim, logging
import bz2
import re
from nltk.tokenize import word_tokenize,sent_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class CorpusIter(object):
	def __init__(self, corpus_path):
		self.corpus_path = corpus_path
	def __iter__(self):
		if self.corpus_path.endswith('txt'):
			with open(self.corpus_path,'r') as f:
				for line in f:
					yield line.lower().split()
		else:
			for dir_name, subdirList, fileList in os.walk(self.corpus_path):
				for fname in fileList:
					filepath=os.path.join(dir_name,fname)
					with bz2.BZ2File(filepath,'r') as f:
						for line in f:
							# Decode bytestring to string
							line=line.decode("utf-8") 
							if line.startswith('<doc id=') or line.startswith('</doc>'):
								continue
							sentences=sent_tokenize(line.strip())
							for sentence in sentences:
								yield word_tokenize(sentence)


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-c','--corpus-path', help='Corpus path', required=True)
	parser.add_argument('-o','--output-path', help='Output folder', required=True)

	args = parser.parse_args()
	corpus_path = args.corpus_path
	output_folder = args.output_path

	# word2vec model config
	params = {
	'size':300, # vector size
	'window':10, # context window
	'min_count':10, # min frequency
	'sg':1, # 1 = skip-gram, 0 = cbow
	'negative':0, # negative sampling, if > 0, negative sampling is used, usually 5 to 20 words can be dropped
	}

	# assume all the corpus in one big file
	# e.g., utf-8_clean_corpusMondeAllThema-v2b.txt
	if os.path.isfile(corpus_path):
		# Set output model name
		model_name = 'word2vec_lemonde__'+'_'.join([k+'='+str(v) for k,v in params.items()])
		outf_path = os.path.join(output_folder,model_name)
		logging.info('Training word2vec model: '+outf_path)
		# Train and save		
		sentences = CorpusIter(corpus_path) # a memory-friendly iterator
		model = gensim.models.Word2Vec(sentences)
		model.save(outf_path)
	# assume extracted wikipedia root directory
	elif os.path.isdir(corpus_path):
		# Set output model name
		model_name = 'word2vec_wikipedia__'+'_'.join([k+'='+str(v) for k,v in params.items()])
		outf_path = os.path.join(output_folder,model_name)
		# Train and save
		logging.info('Training word2vec model: '+outf_path)
		wikipedia = CorpusIter(corpus_path)
		model = gensim.models.Word2Vec(wikipedia, 
										size=params['size'], 
										window=params['window'], 
										min_count=params['min_count'], 
										sg=params['sg']) # train skip-gram