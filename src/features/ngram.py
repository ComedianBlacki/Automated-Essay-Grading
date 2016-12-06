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


# Read in training data
# Note that for essay set 2, score becomes average of 2 domain scores
train_df = get_training_data('../data/training_set_rel3.tsv')
train_df = standardize_training_scores(train_df)

# train_df, standardization_data = standardize_training_scores(train_df)


# print "\nThe regularized data for each essay set = ", standardization_data

print "\nHead of Training Data Frame"
print train_df.head()


# Read in validation data
valid_df = get_validation_data('../data/valid_set.tsv')

# returned a copy of old_df, with essays cleaned for count vectorizer
# cleaning returns essay with only lowercase words separated by space
def vectorizer_clean(old_df):
    new_df = old_df.copy()
    for i in xrange(new_df.shape[0]):
        new_df.set_value(i, 'essay', " ".join(re.sub('[^a-zA-Z\d\s]', '', new_df['essay'].iloc[i]).lower().split())) 
    return new_df

# print essays cleaned for vectorizer (essay is now just lowercase words separated by space) 
vectorizer_train = vectorizer_clean(train_df)
print "\nVectorizer Train head"
print vectorizer_train.head()

# print essays cleaned for vectorizer (essay is now just lowercase words separated by space) 
vectorizer_valid = vectorizer_clean(valid_df)
print "\nVectorizer Valid head"
print vectorizer_valid.head()

######################################
## TfidfVectorizer with ngram=(1,1) ##
######################################

vectorizer = TfidfVectorizer(stop_words = 'english', max_features=100000)


max_essay_set = max(train_df['essay_set'])


def linear_fit_and_score(essay_set):

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

	print np.array(train_vectors).shape
	print np.array(train_std_scores).shape

	# Linear Model
	regr = LinReg(fit_intercept=False, copy_X=False)
	regr.fit(train_vectors, train_std_scores)


	valid_vectors = vectorizer.transform((vectorizer_valid[vectorizer_valid['essay_set'] == essay_set])['essay'].values).toarray()

	# My guess is we will want to denormalize these scores for quadratic weighted k
	valid_pred_std_scores = regr.predict(valid_vectors)


	# # TODO: Append just to correct essay set
	# # Appending predicted scores to validation data set
	# valid_df["linear_predicted_scores"] = valid_pred_std_scores

	# #de-std the values and placing them into the stand_pred_values array
	# stand_pred_values = []
	# for i in range(max_essay_set):
	#     current_set = valid_df[valid_df['essay_set'] == i + 1]['linear_predicted_scores']
	#     for value in current_set:
	#         stand_pred_values.append(int(float(value) * float(standardization_data[i][2]) + (standardization_data[i][1])))
	# # print stand_pred_values

	# #adding the de-std predicted values to the valid_df dataset
	# valid_df['destd_linear_predicted_scores'] = stand_pred_values

	# ###############
	# #   Scoring   #
	# ###############

	# #Scoring the predicted values with the actual values
	# count = 0
	# for i in range(len(valid_df)):
	#     if valid_df.iloc[i]['score'] == valid_df.iloc[i]['destd_linear_predicted_scores']:
	#         count += 1
	        
	# print "LINEAR"
	# print "Number of correct predictions =", count
	# print "Total number of observations =", len(valid_df)
	# print "Score =", float(count) / len(valid_df)

	# Maybe delete?
	# print "Linear:", Spearman(a = valid_df["score"], b = valid_df["linear_predicted_scores"])


	print "Linear for Essay Set "+str(essay_set)+":", Spearman(a = (valid_df[valid_df['essay_set'] == essay_set])["score"], b = valid_pred_std_scores)
	print "\n"


for i in range(1, max_essay_set+1):
	linear_fit_and_score(i)




