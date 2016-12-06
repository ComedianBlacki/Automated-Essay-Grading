# IMPORT NECESSARY LIBRARIES
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import discriminant_analysis as da
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score
from nltk.corpus import wordnet

##################################################################
#################### NUM OF SENTENCES FEATURE ####################
##################################################################

# REGULARIZING FOR SENTENCE LENGTHS 
def append_regularized_sentence_count(old_df):
    new_df = old_df.copy()
    new_df['sentence_count_std'] = new_df.groupby(['essay_set'])[['sentence_count']].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    return new_df

def create_regularization_sentence_count(old_df):
    # getting the number of datasets
    max_essay_set = max(old_df['essay_set'])
    # list of the regularized values
    regularization_data = []
    for i in range(max_essay_set+1):
        mean = np.mean((old_df[old_df['essay_set'] == i + 1])['sentence_count'])
        std = np.std((old_df[old_df['essay_set'] == i + 1])['sentence_count'])
        regularization_data.append([i + 1, mean, std])
    return regularization_data

def sentences(par):
    split_sent = re.split(r'[.!?]+', par)
    return len(split_sent)

def fill_sentence_column(train_df, valid_df):
    numOfSent_train = []
    for essay in train_df['essay']:
        sent = sentences(essay)
        numOfSent_train.append(sent)

    numOfSent_valid = []
    for essay in valid_df['essay']:
        sent = sentences(essay)
        numOfSent_valid.append(sent)

    train_df['sentence_count'] = numOfSent_train
    valid_df['sentence_count'] = numOfSent_valid

    train_df = append_regularized_sentence_count(train_df)
    valid_df = append_regularized_sentence_count(valid_df)
    '''
    # DENORMALIZING FOR THE VALID SET
    regularization_data_sentence = create_regularization_sentence_count(train_df)
    max_essay_set = max(train_df['essay_set'])
    stand_pred_values_l2 = []
    for i in range(max_essay_set):
        current_set = valid_df[valid_df['essay_set'] == i + 1]['sentence_count_std']
        for value in current_set:
            stand_pred_values_l2.append(round(float(value) * float(regularization_data_sentence[i][2]) + (regularization_data_sentence[i][1])))

    valid_df['new_sentence_count'] = stand_pred_values_l2
    
    assert (valid_df['new_sentence_count'] == valid_df['sentence_count'])'''
    return train_df, valid_df
