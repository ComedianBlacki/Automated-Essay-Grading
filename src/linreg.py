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

#COLS = ['std_sentence_count', 'spelling_correct', 'std_unique_words', 'std_total_words', 'essay_set', 'std_total_words', 'std_unique_words'
#		'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X', 'std_perplexity', 'std_score']

#COLS = ['std_sentence_count', 'essay_set', 'std_score']
#train_df = train_df[COLS].join(train_df.filter(regex=("tfidf_*")))
#valid_df = valid_df[COLS].join(valid_df.filter(regex=("tfidf_*")))

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

	#print "Linear for Essay Set "+str(i)+":", Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores)
	#print "\n"

	alpha = [x*1.0/20 for x in range(21)]
	ridge_scores = []
	lasso_scores = []
	for a in alpha:
		ridge = linear_model.Ridge(alpha = a)
		ridge.fit(train_x, train_std_scores)
		valid_pred_std_scores_ridge = ridge.predict(valid_x)

		new_ridge_score = Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores_ridge)[0]
		ridge_scores.append(new_ridge_score)

		lasso = linear_model.Lasso(alpha = a)
		lasso.fit(train_x, train_std_scores)
		valid_pred_std_scores_lasso = lasso.predict(valid_x)
		new_lasso_score = Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores_ridge)[0]
		lasso_scores.append(new_ridge_score)

	best_score_ridge = np.max(ridge_scores)
	best_alpha_ridge = alpha[np.argmax(ridge_scores)]
	print "Linear (RIDGE alpha=" + str(best_alpha_ridge) +") for Essay Set "+str(i)+":", str(best_score_ridge)
	print "\n"

	best_score_lasso = np.max(lasso_scores)
	best_alpha_lasso = alpha[np.argmax(lasso_scores)]
	print "Linear (LASSO alpha =" + str(best_alpha_lasso) + ") for Essay Set "+str(i)+":", str(best_score_lasso)
	print "\n"