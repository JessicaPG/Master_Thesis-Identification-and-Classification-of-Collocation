"""multiclass_SVM.py: trained a multiclass classifier with SVM"""
__author__ = "Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"

# -*- coding: utf-8 -*-
import gensim
import numpy as np
from sklearn import svm
import os
from argparse import ArgumentParser
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import classification_report

# Global variables
USE_SEVEN = False
composition = ['concat', 'diff', 'sum', 'mult']


def data_processed(inp_dir, composition, data_type='txt'):
    """
    Process and transforms the data for fitting it into a classifier
    :param inp_dir: data input directory
    :param model: word2vec embeddings
    :param composition: type of composition to forming the colocation embedding (concat, add, subs, mult)
    :param data_type: expected .txt
    :return: colEmbeddings (samples), label, and indx2word (mapping collocation with class)
    """

    colEmbeddings = []
    indx2word = {}
    label = []
    index = 0
    listing=[f for f in os.listdir(inp_dir) if f.endswith(data_type)]
    for infile in listing:
        class_label = infile.split('.txt')[0]

        with open(os.path.join(inp_dir,infile) ,'r') as f:
            for line in f:
                index += 1

                line = line.rstrip('\n').split('\t')
                base = line[0]
                col = line[1]
                rel = base + '__' + col

                if base in model.wv.vocab and col in model.wv.vocab:
                    if composition == 'concat':
                        xi = np.concatenate((model[base],model[col]))

                    elif composition == 'diff':
                        xi = (model[base] - model[col])

                    elif composition == 'sum':
                        xi = (model[base] + model[col])

                    elif composition == 'mult':
                        xi = (model[base] * model[col])

                    if USE_SEVEN:
                        if rel in modelS.wv.vocab:
                            xi = np.concatenate([xi, modelS[rel]])
                        else:
                            xi = np.concatenate([xi, np.zeros(modelS.vector_size)])

                    colEmbeddings.append(xi)
                    label.append(class_label)
                    indx2word[index] = class_label

    return colEmbeddings,label, indx2word


def load_embeddings(embeddings_path):
    print('Loading embeddings:',embeddings_path)
    try:
        model=gensim.models.Word2Vec.load(embeddings_path)
    except:
        try:
            model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
        except:
            try:
                model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
            except:
                sys.exit('Couldnt load embeddings')

    return model


def plot_confusion_matrix(Y_test, predicted,labels):
    "Plot confusion matrix"
    cm = confusion_matrix(Y_test, predicted, labels)
    normalized = cm / cm.astype(np.float).sum(axis=1, keepdims=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(normalized)
    plt.title('Confusion matrix of composition: ' + elem)
    fig.colorbar(cax)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def print_results (scores):
    table = PrettyTable()
    table.field_names = ["Operation", "Accuracy", "Precision", "Recall", "F1"]
    table.add_row(["Concat", scores[0][0], scores[0][1], scores[0][2], scores[0][3]])
    table.add_row(["Diff", scores[1][0], scores[1][1], scores[1][2], scores[1][3]])
    table.add_row(["Add", scores[2][0], scores[2][1], scores[2][2], scores[2][3]])
    table.add_row(["Mult", scores[3][0], scores[3][1], scores[3][2], scores[3][3]])
    print(table)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-i', '--train_dir', help='training samples dir', required=True)
    parser.add_argument('-t', '--test_dir', help='test samples dir', required=True)
    parser.add_argument('-wv', '--embeddings', help='word2vec emb', required=True)
    parser.add_argument('-swv', '--seven', help='relation embeddings', required=True)

    args = parser.parse_args()
    train_data = args.train_dir
    test_data = args.test_dir
    embeddings = args.embeddings
    sev_emb = args.seven

    scores =[]
    det_report =[]
    # Load word2vec and process data
    #model = model = gensim.models.Word2Vec.load(embeddings)
    model = load_embeddings(embeddings)
    modelS = load_embeddings(sev_emb)

    for elem in composition:
        X_train, Y_train, indexes_train = data_processed(train_data, elem)
        X_test, Y_test, indexes_test = data_processed(test_data, elem)


        # Train the multiclass SVM model
        clf = svm.SVC(gamma='scale')
        clf.fit(X_train, Y_train)

        # Testing the classifier
        predicted = clf.predict(X_test)
        mapping = list(indexes_test.values())


        #for i in range(len(X_test)):
         #   print("X = %s, Predicted = %s" % (mapping[i], predicted[i]))

        #labels = ['antibon', 'antimagn', 'bon', 'causfunc0', 'liqufunc0', 'magn', 'oper1', 'real1', 'sing','noise']
        labels = ['cause', 'experimenter', 'intensity', 'manifest', 'noise', 'phase']

        scores.append((accuracy_score(Y_test, predicted), precision_score(Y_test, predicted, average='weighted'), recall_score(Y_test, predicted,average='weighted'),
                      f1_score(Y_test, predicted, average='weighted')))

        plot_confusion_matrix(Y_test, predicted,labels)
        det_report.append(classification_report(Y_test, predicted, target_names=labels))

    print_results(scores)
    for x in range(len(det_report)):
        print("Detail report of composition ", composition[x])
        print(det_report[x])



