import matplotlib
import matplotlib.pyplot as plt
import features.util as util

from features.pos_tags import *
from features.spelling import *
from features.sentences import *
from features.perplexity import Perplexity
from features.tfidf import *
from features.unique_words import *
from features.total_words import *

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.linear_model import LinearRegression as LinReg
from scipy.stats import spearmanr as Spearman
from sklearn import linear_model
import pickle

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

print "Fetching data..."
train_df = util.get_training_data('../data/training_set_rel3.tsv')
valid_df = util.get_validation_data('../data/valid_set.tsv')

print "Standardizing scores..."
train_df, valid_df = util.append_standardized_column(train_df, valid_df, 'score')   

print "Calculating perplexity feature..."
train_df, valid_df = Perplexity().fill_perplexity_columns(train_df, valid_df)

print "Calculating number of sentences feature..."
train_df, valid_df = fill_sentence_column(train_df, valid_df)

print "Cleaning for spelling and word count..."
# cleaned up data for spelling feature
vectorizer_train_spelling = util.vectorizer_clean_spelling(train_df)
train_essays_spelling = vectorizer_train_spelling['essay'].values
vectorizer_valid_spelling = util.vectorizer_clean_spelling(valid_df)
valid_essays_spelling = vectorizer_valid_spelling['essay'].values

print "Calculating total words feature..."
train_df, valid_df = fill_total_words_column(train_df, valid_df, train_essays_spelling, valid_essays_spelling)

print "Calculating unique words feature..."
train_df, valid_df = fill_unique_words_column(train_df, valid_df, train_essays_spelling, valid_essays_spelling)

print "Calculating spelling feature..."
train_df, valid_df = fill_spelling_column(train_df, valid_df, train_essays_spelling, valid_essays_spelling)


# plot essay set number 1 word count versus score
features = ['std_perplexity', 'spelling_correct', 'std_sentence_count', 'std_total_words', 'std_unique_words']
titles = ['Perplexity vs Score', 'Correct Spelling vs Score', 'Sentence Count vs Score', 'Total Words vs Score', \
        'Unique Words vs Score']
#features = ['std_perplexity', 'std_sentence_count']
#titles = ['Perplexity vs Score', 'Sentenc Ccount vs Score'] 

#features = ['spelling_correct', 'std_total_words', 'std_unique_words']
#titles = ['Spelling vs Score', 'Total Words vs Score', 'Unique Words vs Score'] 

axes = [ax1, ax2, ax3, ax4, ax5]
#axes = [ax1, ax2, ax3]
for feature, title, ax in zip(features, titles, axes):
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel('Standardized Score')
    ax.scatter(train_df[feature].values, train_df['std_score'].values)

fig.tight_layout()
plt.show()
