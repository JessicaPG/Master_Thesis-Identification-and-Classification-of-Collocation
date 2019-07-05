"""Split_data.py: Split data into train and test subset (80/20%)"""
__author__ = "Jessica Pérez Guijarro"
__email__ =  "jessicaperezgui@gmail.com"

import os
from sklearn.model_selection import train_test_split



def data_processed(inp_dir, data_type='txt'):
    train_path = inp_dir + '/train'
    test_path = inp_dir + '/test'

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
            os.makedirs(test_path)

    listing=[f for f in os.listdir(inp_dir) if f.endswith(data_type)]



    for infile in listing:
        class_label = infile.split('.txt')[0]
        temp_arr = []
        with open(os.path.join(inp_dir,infile) ,'r') as f:
            for line in f:
                temp_arr.append(line)
            X_train, X_test = train_test_split(temp_arr,test_size=0.20, random_state=42)

            for pair in X_train:
                with  open(os.path.join(train_path, class_label +'.txt'), 'a') as trainF:
                    trainF.write(pair)

            for pair in X_test:
                with  open(os.path.join(test_path, class_label +'.txt'), 'a') as testF:
                    testF.write(pair)

data_processed('/Users/jessie/PycharmProjects/thesis/data/ES/all')