import util

from features.pos_tags import *

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

	# Should go after all features are filled in
	train_df = util.move_column_last(train_df, 'score')
	train_df = util.move_column_last(train_df, 'std_score')

	valid_df = util.move_column_last(valid_df, 'score')

	print train_df.head()

	print valid_df.head()

	# implement model here


if __name__ == "__main__": main()