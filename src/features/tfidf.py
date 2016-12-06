import pandas as pd
import numpy as np
import re
import util
import string
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer

def column(matrix, i):
    return [row[i] for row in matrix]

def normalize_tfidf_column(matrix):
	normed_matrix = normalize(matrix, axis=1, norm='l1')
	return normed_matrix

def fill_tfidf_column(train_df, valid_df, train_essays, valid_essays, ngrams):
	vectorizer = TfidfVectorizer(stop_words = 'english', max_features=10, ngram_range=(ngrams, ngrams))
	train_vectors = vectorizer.fit_transform(train_essays).toarray()
	valid_vectors = vectorizer.fit_transform(valid_essays).toarray()

	train_vectors_norm = normalize_tfidf_column(train_vectors)
	valid_vectors_norm = normalize_tfidf_column(valid_vectors)
			
	for i in range(len(train_vectors_norm[0])):
		new_column = column(train_vectors_norm, i)
		label = str(ngrams)+'-gram_tfidf_' + str(i+1)
		new_df = pd.DataFrame({label: new_column})
		train_df = train_df.join(new_df)

	for i in range(len(valid_vectors_norm[0])):
		new_column = column(valid_vectors_norm, i)
		label = str(ngrams)+'-gram_tfidf_' + str(i+1)
		#print label
		new_df = pd.DataFrame({label: new_column})
		valid_df = valid_df.join(new_df)

	return train_df, valid_df



