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

# NOTE: FEEL FREE TO COMMENT OUT LINES TO SPEED UP YOUR TESTING
def main():

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
	# spelling feature
	train_df, valid_df = fill_spelling_column(train_df, valid_df, train_essays_spelling, valid_essays_spelling)
	
	print "Calculating pos tags features..."

	train_df, valid_df = fill_pos_columns(train_df, valid_df)

	print "Cleaning for TFIDF..."
	# cleaned up data for tfidf vector feature
	vectorizer_train = util.vectorizer_clean(train_df)
	train_essays = vectorizer_train['essay'].values
	vectorizer_valid = util.vectorizer_clean(valid_df)
	valid_essays = vectorizer_valid['essay'].values

	print "Calculating TFIDF features with unigram..."

	# tfidf vector feature with unigram
	train_df, valid_df = fill_tfidf_column(train_df, valid_df, train_essays, valid_essays, 1)
	#print "Calculating TFIDF features with bigram..."

	# tfidf vector feature with unigram
	#train_df, valid_df = fill_tfidf_column(train_df, valid_df, train_essays, valid_essays, 2)

	#print "Calculating TFIDF features with trigram..."

	# tfidf vector feature with unigram
	#train_df, valid_df = fill_tfidf_column(train_df, valid_df, train_essays, valid_essays, 3)
    
	#print "Moving scores to right end of dataframe"

	# Should go after all features are filled in
	#train_df = util.move_column_last(train_df, 'score')
	#train_df = util.move_column_last(train_df, 'std_score')
	#valid_df = util.move_column_last(valid_df, 'score')

	print train_df.head()

	print valid_df.head()

	COLS = ['essay_set', 'spelling_correct', 'std_sentence_count', 'std_unique_words', 'std_total_words', 'std_unique_words',
		'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X', 'std_perplexity', 'std_score']

	train_df = train_df[COLS].join(train_df.filter(regex=("tfidf_*")))
	valid_df = valid_df[COLS].join(valid_df.filter(regex=("tfidf_*")))

	print train_df.shape
	print valid_df.shape

	"""
	train_df.to_hdf('train.h5','df')
	valid_df.to_hdf('valid.h5','df')

	train_df = None
	valid_df = None

	train_df = pd.read_hdf('train.h5', 'df')
	valid_df = pd.read_hdf('valid.h5', 'df')
	"""

	max_essay_set = max(train_df['essay_set'])

	for i in range(1, max_essay_set+1):

		print ""

		train_x = np.asarray((train_df[train_df['essay_set'] == i]).drop(['essay_set', 'std_score'], axis=1))
		train_std_scores = np.asarray((train_df[train_df['essay_set'] == i])['std_score'], dtype="|S6").astype(np.float)

		regr = LinReg(fit_intercept=False, copy_X=False)
		regr.fit(train_x, train_std_scores)

		valid_x = np.asarray((valid_df[valid_df['essay_set'] == i]).drop(['essay_set', 'std_score'], axis=1))
		valid_pred_std_scores = regr.predict(valid_x)

		print "Linear for Essay Set "+str(i)+":", Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores)

		alpha = [x*1.0/20 for x in range(20, 0, -1)]
		ridge_scores = []
		lasso_scores = []
		for a in alpha:
			ridge = linear_model.Ridge(alpha = a)
			ridge.fit(train_x, train_std_scores)
			valid_pred_std_scores_ridge = ridge.predict(valid_x)

			new_ridge_score = Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores_ridge)[0]
			ridge_scores.append(new_ridge_score)

			print "Alpha = " + str(alpha) + " Ridge for Essay Set "+str(i)+":", new_ridge_score

			lasso = linear_model.Lasso(alpha = a)
			lasso.fit(train_x, train_std_scores)
			valid_pred_std_scores_lasso = lasso.predict(valid_x)

			new_lasso_score = Spearman(a = (valid_df[valid_df['essay_set'] == i])["std_score"], b = valid_pred_std_scores_lasso)[0]
			lasso_scores.append(new_lasso_score)

			print "Alpha = " + str(alpha) + "Lasso for Essay Set "+str(i)+":", new_lasso_score

		print ""

		best_score_ridge = np.max(ridge_scores)
		best_alpha_ridge = alpha[np.argmax(ridge_scores)]
		print "Linear (RIDGE alpha=" + str(best_alpha_ridge) +") for Essay Set "+str(i)+":", str(best_score_ridge)
		print ""

		best_score_lasso = np.max(lasso_scores)
		best_alpha_lasso = alpha[np.argmax(lasso_scores)]
		print "Linear (LASSO alpha =" + str(best_alpha_lasso) + ") for Essay Set "+str(i)+":", str(best_score_lasso)
		print ""

if __name__ == "__main__": main()