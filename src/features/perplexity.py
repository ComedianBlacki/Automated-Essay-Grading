import math
import util
from nltk import trigrams
from nltk import bigrams
from nltk import everygrams
from collections import Counter

def fill_perplexity_columns(train_df, valid_df):
	train_clean = util.vectorizer_clean(train_df)
	valid_clean = util.vectorizer_clean(valid_df)

	print "Creating ngram counts..."

	counts = create_counts(train_clean['essay'].values)

	dfs = [train_clean, valid_clean]

	for j, df in enumerate(dfs):
		for i in xrange(df.shape[0]):
			if i % 1000 == 0:
				essay_set = None
				if j == 0:
					essay_set = "Train"
				else:
					essay_set = "Validation"
				
				print essay_set + " essay " + str(i) + " of " + str(df.shape[0])
			essay = df.get_value(i, 'essay')
			perp = perplexity(counts, essay)

			df = df.set_value(i, 'perplexity', perp)

	train_df['perplexity'] = train_clean['perplexity']
	valid_df['perplexity'] = valid_clean['perplexity']

	return util.append_standardized_column(train_df, valid_df, 'perplexity')

# apply LaPlace smoothing, incrementing all counts by 1
class LaPlaceCounter(Counter):
	def __getitem__(self, idx):
		if idx in self.keys():
			return dict.__getitem__(self, idx)
		else:
			return 1

# train_essays must be cleaned before being passed in
def create_counts(train_essays):
	unigram_counts = LaPlaceCounter()
	for i, essay in enumerate(train_essays):
		if i % 1000 == 0:
			print "Essay " + str(i) + " of " + str(len(train_essays))
		word_list = essay.split()
		for word in word_list:
			unigram_counts[word] += 1
	return unigram_counts

# After having already fit model on a set of training essays, calculates the
# perplexity of a student's essay based from the model, and returns this
# perplexity to be used as a feature
def perplexity(counts, test_essay):
	log_prob = 0.0
	word_list = test_essay.split()
	num_words = sum(counts.values())
	for word in word_list:
		log_prob += math.log(float(counts[word]) / num_words)
	return math.pow(2.0, -log_prob / len(word_list))

def main():
	train_essays = [
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	]

	test_essay = "what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color what is your favorite color "
	test_essay2 = "my name is annie kevin"
	test_essay3 = "hi my name is kevin"
	test_essay4 = "blah blah blah blah blah"
	test_essay5 = "hi"
	test_essay6 = "hi my name is"

	counts = create_counts(train_essays)

	print perplexity(counts, test_essay)
	print perplexity(counts, test_essay2)
	print perplexity(counts, test_essay3)
	print perplexity(counts, test_essay4)
	print perplexity(counts, test_essay5)
	print perplexity(counts, test_essay6)

if __name__ == "__main__": main()
