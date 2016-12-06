from util import *

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LinearRegression as LinReg
# from sklearn.linear_model import LogisticRegressionCV as LogRegCV
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr as Spearman


# Prints head of df
def head_of_df(df, name):
	print "\n"+name
	print df.head()

# Read in training data
# Note that for essay set 2, score becomes average of 2 domain scores
train_df = get_training_data('../../data/training_set_rel3.tsv')
valid_df = get_validation_data('../../data/valid_set.tsv')

train_df, valid_df = append_standardized_column(train_df, valid_df, 'score')

head_of_df(train_df, 'Head of Training Data Frame')

# Returns a copy of old_df, with essays cleaned for count vectorizer
# Cleaning returns essay with only lowercase words separated by space
def vectorizer_clean(old_df):
    new_df = old_df.copy()
    for i in xrange(new_df.shape[0]):
        new_df.set_value(i, 'essay', " ".join(re.sub('[^a-zA-Z\d\s]', '', new_df['essay'].iloc[i]).lower().split())) 
    return new_df

vectorizer_train = vectorizer_clean(train_df)
vectorizer_valid = vectorizer_clean(valid_df)

head_of_df(vectorizer_train, 'Vectorizer Train head')
head_of_df(vectorizer_valid, 'Vectorizer Valid head')

max_essay_set = max(train_df['essay_set'])


def linear_fit_and_score(essay_set, vectorizer, name):

	#Get all the text from data
	train_essays = (vectorizer_train[vectorizer_train['essay_set'] == essay_set])['essay'].values

	#Turn each text into an array of word counts
	train_vectors = vectorizer.fit_transform(train_essays).toarray()

	#Standardizing for y
	train_std_scores = np.asarray((vectorizer_train[vectorizer_train['essay_set'] == essay_set])['std_score'], dtype="|S6").astype(np.float)
	# print "\nStandardized Train Scores", train_std_scores[:5]

	##############
	#   Linear   #
	##############

	# Linear Model
	regr = LinReg(fit_intercept=False, copy_X=False)
	regr.fit(train_vectors, train_std_scores)

	valid_vectors = vectorizer.transform((vectorizer_valid[vectorizer_valid['essay_set'] == essay_set])['essay'].values).toarray()

	# My guess is we will want to denormalize these scores for quadratic weighted k
	valid_pred_std_scores = regr.predict(valid_vectors)

	print "Linear for Essay Set "+str(essay_set)+" with "+name+":", Spearman(a = (valid_df[valid_df['essay_set'] == essay_set])["score"], b = valid_pred_std_scores)
	print "\n"

# TfidfVectorizer with ngram=(1,1)
vectorizer1 = TfidfVectorizer(stop_words = 'english', max_features=100000, ngram_range=(1, 1))

# TfidfVectorizer with ngram=(2,2)
vectorizer2 = TfidfVectorizer(stop_words = 'english', max_features=100000, ngram_range=(2, 2))

# TfidfVectorizer with ngram=(3,3)
vectorizer3 = TfidfVectorizer(stop_words = 'english', max_features=100000, ngram_range=(3, 3))

vectorizers = [(vectorizer1, "ngram=(1,1)"), (vectorizer2, "ngram=(2,2)"), (vectorizer3, "ngram=(3,3)")]

for vectorizer, name in vectorizers:
	for i in range(1, max_essay_set+1):
		linear_fit_and_score(i, vectorizer, name)




