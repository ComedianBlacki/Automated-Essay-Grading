import math
import util
from nltk import trigrams
from nltk import bigrams
from nltk import everygrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# class that implements functionality for calculating perplexity of essay with Laplace smoothing
class Perplexity:
	def __init__(self):
   		self.num_words = None
		self.counts = None
		self.vectorizer = None

	def create_counts(self, compressed_essays):
		self.vectorizer = CountVectorizer().fit(compressed_essays)
		self.counts = self.vectorizer.transform(compressed_essays).toarray()[0]

		# length added for LaPlace smoothing
		self.num_words = float(sum(self.counts) + len(self.counts))

	def fill_perplexity_columns(self, train_df, valid_df):
		print "Creating ngram COUNTS..."
		self.create_counts(util.perplexity_clean(train_df))

		train_clean = util.vectorizer_clean(train_df)
		valid_clean = util.vectorizer_clean(valid_df)

		dfs = [train_clean, valid_clean]

		for j, df in enumerate(dfs):
			for i in xrange(df.shape[0]):
				if i % 20 == 0:
					essay_set = None
					if j == 0:
						essay_set = "Train"
					else:
						essay_set = "Validation"
					
					print essay_set + " essay " + str(i) + " of " + str(df.shape[0])
				essay = df.get_value(i, 'essay')
				perp = self.perplexity(essay)

				df = df.set_value(i, 'perplexity', perp)

		train_df['perplexity'] = train_clean['perplexity']
		valid_df['perplexity'] = valid_clean['perplexity']

		return util.append_standardized_column(train_df, valid_df, 'perplexity')

	# After having already fit model on a set of training essays, calculates the
	# perplexity of a student's essay based from the model, and returns this
	# perplexity to be used as a feature
	def perplexity(self, test_essay):
		log_prob = 0.0
		word_list = test_essay.split()
		for word in word_list:
			if word in self.vectorizer.vocabulary_:
				log_prob += math.log( (self.counts[self.vectorizer.vocabulary_[word]] + 1.0) / self.num_words)
			else:
				log_prob += math.log (1.0 / self.num_words)

		return math.pow(2.0, -log_prob / len(word_list))

def main():
	train_essays = ["my name is kevin hi my name is kevin hi my name is kevin hi my name is kevin hi my name is kevin hi my name is kevin"]

	test_essay = "what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color "
	test_essay2 = "my name is annie kevin"
	test_essay3 = "hi my name is kevin"
	test_essay4 = "blah blah blah blah blah"
	test_essay5 = "hi"
	test_essay6 = "hi my name is"

	perp = Perplexity()

	counts = perp.create_counts(train_essays)

	print perp.vectorizer.vocabulary_

	print perp.perplexity(test_essay)

	print perp.vectorizer.vocabulary_

	print perp.perplexity(test_essay2)
	print perp.perplexity(test_essay3)
	print perp.perplexity(test_essay4)
	print perp.perplexity(test_essay5)
	print perp.perplexity(test_essay6)

	print perp.vectorizer.vocabulary_
	print perp.counts
	print perp.num_words
	print perp.counts[perp.vectorizer.vocabulary_['hi']]

if __name__ == "__main__": main()
