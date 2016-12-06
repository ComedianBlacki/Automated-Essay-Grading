import pandas as pd
import numpy as np
import re
import util
import string

from sklearn.feature_extraction.text import TfidfVectorizer

def column(matrix, i):
    return [row[i] for row in matrix]

def word_count(essay):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', essay)
    return len(out.split())

def normalize_tfidf_column(essays, vectors):
	vectors_norm = []
	for idx, essay in enumerate(essays):
		num_of_words = word_count(essay)
		new_vectors = []
		for idx2 in range(vectors):
			normalized_value = 1. * vectors[idx][idx2] / num_of_words
			new_vectors.append(normalized_value)
		vectors_norm.append(new_vectors)
	return vectors_norm

def fill_tfidf_column(train_df, valid_df, train_essays, valid_essays):
	vectorizer = TfidfVectorizer(stop_words = 'english', max_features=1)
	train_vectors = vectorizer.fit_transform(train_essays).toarray()
	valid_vectors = vectorizer.fit_transform(valid_essays).toarray()

	train_vectors_norm = normalize_tfidf_column(train_essays, train_vectors)
	valid_vectors_norm = normalize_tfidf_column(valid_essays, valid_vectors)
			
	for i in range(len(train_vectors_norm[0])):
		new_column = column(train_vectors_norm, i)
		label = 'tfidf_' + str(i+1)
		new_df = pd.DataFrame({label: new_column})
		train_df = train_df.join(new_df)

	for i in range(len(valid_vectors_norm[0])):
		new_column = column(valid_vectors_norm, i)
		label = 'tfidf_' + str(i+1)
		#print label
		new_df = pd.DataFrame({label: new_column})
		valid_df = valid_df.join(new_df)

	return train_df, valid_df




