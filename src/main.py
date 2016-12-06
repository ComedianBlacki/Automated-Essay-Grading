import util

from features.pos_tags import *
from features.spelling import *
from features.sentences import *
from features.tfidf_3gram import *

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV

def main():
	train_df = util.get_training_data('../data/training_set_rel3.tsv')
	train_df = util.standardize_training_scores(train_df)
	valid_df = util.get_validation_data('../data/valid_set.tsv')


	train_df, valid_df = fill_pos_columns(train_df, valid_df)
	train_df, valid_df = fill_sentence_colum(train_df, valid_df)

	# cleaned up data for spelling feature
	vectorizer_train_spelling = util.vectorizer_clean_spelling(train_df)
	train_essays_spelling = vectorizer_train_spelling['essay'].values
	vectorizer_valid_spelling = util.vectorizer_clean_spelling(valid_df)
	valid_essays_spelling = vectorizer_valid_spelling['essay'].values

	# spelling feature
	train_df, valid_df = fill_spelling_column(train_df, valid_df, train_essays_spelling, valid_essays_spelling)

	# cleaned up data for tfidf vector feature
	vectorizer_train = util.vectorizer_clean(train_df)
	train_essays = vectorizer_train['essay'].values
	vectorizer_valid = util.vectorizer_clean(valid_df)
	valid_essays = vectorizer_valid['essay'].values

	# tfidf vector feature with unigram
	train_df, valid_df = fill_tfidf_column(train_df, valid_df, train_essays, valid_essays)

	# Should go after all features are filled in
	train_df = util.move_column_last(train_df, 'score')
	train_df = util.move_column_last(train_df, 'std_score')

	valid_df = util.move_column_last(valid_df, 'score')

	print train_df.head()

	print valid_df.head()

	# implement model here


if __name__ == "__main__": main()