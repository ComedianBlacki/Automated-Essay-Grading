import features.util as util

from features.pos_tags import *
from features.spelling import *
from features.sentences import *
from features.perplexity import *
from features.tfidf_3gram import *

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.linear_model import LinearRegression as LinReg
from scipy.stats import spearmanr as Spearman
from sklearn import linear_model
import pickle

# Use cross-validation to regularize the linear regression model

train_df = pd.read_pickle('train_df.txt')
valid_df = pd.read_pickle('valid_df.txt')


max_essay_set = max(train_df['essay_set'])

COLS = ['std_sentence_count', 'spelling_correct', 'std_unique_words', 'std_total_words', 
		'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X', 'std_perplexity', 'std_score']

train_df = train_df[COLS].join(train_df.filter(regex=("tfidf_*")))
valid_df = valid_df[COLS].join(valid_df.filter(regex=("tfidf_*")))

for i in range(1, max_essay_set+1):

	#vectorizer_train = util.vectorizer_clean(train_df)
	train_x = np.asarray((train_df[train_df['essay_set'] == i]).drop('std_score', axis=1))
	#train_x = np.asarray((train_df[train_df['essay_set'] == i])[['std_sentence_count']])
	train_std_scores = np.asarray((train_df[train_df['essay_set'] == i])['std_score'], dtype="|S6").astype(np.float)
	

	regr = LinReg(fit_intercept=False, copy_X=False)
	regr.fit(train_x, train_std_scores)

	valid_x = np.asarray((valid_df[valid_df['essay_set'] == i]).drop('std_score', axis=1))
	#valid_x = np.asarray((valid_df[valid_df['essay_set'] == i])[['std_sentence_count']])
	valid_pred_std_scores = regr.predict(valid_x)

	print "Linear for Essay Set "+str(i)+":", Spearman(a = (valid_df[valid_df['essay_set'] == i])["score"], b = valid_pred_std_scores)
	print "\n"

	ridge = linear_model.Ridge(alpha = 0.5)
	ridge.fit(train_x, train_std_scores)
	valid_pred_std_scores_ridge = ridge.predict(valid_x)

	print "Linear (RIDGE) for Essay Set "+str(i)+":", Spearman(a = (valid_df[valid_df['essay_set'] == i])["score"], b = valid_pred_std_scores_ridge)
	print "\n"

	lasso = linear_model.Lasso(alpha = 0.5)
	lasso.fit(train_x, train_std_scores)
	valid_pred_std_scores_lasso = lasso.predict(valid_x)

	print "Linear (LASSO) for Essay Set "+str(i)+":", Spearman(a = (valid_df[valid_df['essay_set'] == i])["score"], b = valid_pred_std_scores_lasso)
	print "\n"