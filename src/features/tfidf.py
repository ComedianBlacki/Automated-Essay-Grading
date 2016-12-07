import pandas as pd
import numpy as np
import re
import util
import string
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 100000

def fill_tfidf_column(train_df, valid_df, train_essays, valid_essays, ngrams):
	vectorizer = TfidfVectorizer(stop_words = 'english', max_features=MAX_FEATURES, ngram_range=(ngrams, ngrams))

	print "Fitting training essays..."

	train_vectors = vectorizer.fit_transform(train_essays).toarray()

	print "Transforming validation essays..."

	valid_vectors = vectorizer.transform(valid_essays).toarray()

	print "Normalizing train vectors..."

	train_vectors = normalize(train_vectors, axis=1, norm='l1')

	print "Normalizing validation vectors..."

	valid_vectors = normalize(valid_vectors, axis=1, norm='l1')

	NUM_COL = len(train_vectors[0])

	col_names = [str(ngrams)+"-gram_tfidf_" + str(i+1) for i in range(NUM_COL)]

	print "Joining train..."

	train_df = train_df.join(pd.DataFrame(train_vectors, index=train_df.index, columns=col_names))

	print "Joining validation..."
	valid_df = valid_df.join(pd.DataFrame(valid_vectors, index=valid_df.index, columns=col_names))

	return train_df, valid_df



