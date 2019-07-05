# -*- coding: utf-8 -*-
"""negative_examples_generator.py: Creates negative samples"""
__author__ = "Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"

import numpy as np
import os
from itertools import product
from argparse import ArgumentParser


def gather_data(inp_dir, data_type='txt'):
    listing = [f for f in os.listdir(inp_dir) if f.endswith(data_type)]

    for infile in listing:
        with open(os.path.join(inp_dir, infile), 'r') as f:
            for pair in f:
                samples.append(pair)


def random_strategy(candidates_dir):
    """Pick random collocation candidates from Wikipedia Corpus"""
    listing = [f for f in os.listdir(candidates_dir) if f.endswith('.txt')]
    cand_samples = []

    for infile in listing:
        with open(os.path.join(candidates_dir, infile), 'r') as f:
            for pair in f:
                cand_samples.append(pair)

    arr = np.random.choice(cand_samples,len(samples)//2)
    for pair in arr:
        pair = pair.rstrip('\n').split('\t')
        data.append(pair[0] + "\t" + pair[1] + "\n")



def cartesian_product_strategy(samples):
    """Cartesian product of pair of words
        input:                  output:
        steak cut               steak cite
        example cite            example cut"""

    base = []
    col = []

    for pair in samples:
        arr = pair.rstrip('\n').split('\t')
        base.append(arr[0])
        col.append(arr[1])

    # Apply cartesian product
    prod = list(product(*[base,col]))

    # iterate over indexes of prod array
    for index in np.random.choice(len(prod), len(samples)//2):
        data.append(prod[index][0] + "\t" + prod[index][1] + "\n")

def opposite_strategy(samples):
    """Transforms a positive instance into its opposite
        heavy rain --> rain heavy """

    arr = np.random.choice(samples,(len(samples) // 3))
    for pair in arr:
        pair = pair.rstrip('\n').split('\t')
        data.append(pair[1] + "\t" + pair[0] + "\n")


def write_file(data, dir, size):
    """ Write samples into a file"""

    with open(os.path.join(dir,'noise.txt'), 'w') as outf:
        for elem in np.random.choice(data, (len(data) * size)//100):
            outf.write(elem)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-i', '--train_dir', help='training samples dir', required=True)
    parser.add_argument('-t', '--test_dir', help='test samples dir', required=True)
    parser.add_argument('-c', '--candidates_dir', help='col candidates dir', required=True)

    args = parser.parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    candidates_dir = args.candidates_dir

    samples = []
    data = []
    gather_data(train_dir)
    gather_data(test_dir)

    # Execute strategies
    random_strategy(candidates_dir)
    cartesian_product_strategy(samples)

    write_file(data,train_dir,80)
    write_file(data, test_dir, 20)
