import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

def column(matrix, i):
    return [row[i] for row in matrix]

def fill_tfidf_column(train_df, valid_df, train_essays, valid_essays):
	vectorizer = TfidfVectorizer(stop_words = 'english', max_features=10)
	train_vectors = vectorizer.fit_transform(train_essays).toarray()
	valid_vectors = vectorizer.fit_transform(valid_essays).toarray()

	for i in range(len(train_vectors[0])):
		new_column = column(train_vectors, i)
		label = 'tfidf_' + str(i+1)
		new_df = pd.DataFrame({label: new_column})
		train_df = train_df.join(new_df)

	for i in range(len(valid_vectors[0])):
		new_column = column(valid_vectors, i)
		label = 'tfidf_' + str(i+1)
		#print label
		new_df = pd.DataFrame({label: new_column})
		valid_df = valid_df.join(new_df)

	return train_df, valid_df




