import math
import util

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
class LaPlaceCounter(dict):
	def __getitem__(self, idx):
		if idx in self.keys():
			return dict.__getitem__(self, idx)
		else:
			return 1

# with thanks and credit to Scott Triglia
# http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# train_essays must
def create_counts(train_essays):
	n_gram_counts = LaPlaceCounter()
	for essay in train_essays:
		word_list = essay.split()
		for i in xrange(1, 4):
			ngrams = find_ngrams(word_list, i)
			for ngram in ngrams:
				n_gram_counts[ngram] += 1
	return n_gram_counts

# After having already fit model on a set of training essays, calculates the
# perplexity of a student's essay based from the model, and returns this
# perplexity to be used as a feature
def perplexity(counts, test_essay):
	log_prob = 0.0
	word_list = test_essay.split()
	tri_grams = find_ngrams(word_list, 3)
	for tri_gram in tri_grams:
		# p(w_3 | w_2, w_1) = p(w_3, w_2, w_1) / p(w_2, w_1)
		log_prob += math.log(float(counts[tri_gram]) / counts[(tri_gram[0], tri_gram[1])])

	# handle when essays are shorter than 3 words
	first = None
	if len(tri_grams) > 0:
		first = tri_grams[0]
	else:
		bi_grams = find_ngrams(word_list, 2)
		if len(bi_grams) > 0:
			first = bi_grams[0]
		else:
			uni_grams = find_ngrams(word_list, 1)
			first = uni_grams[0]

	# handle edge case, where probability calc involves fewer than 3 words
	if len(first) > 1:
		log_prob += math.log(float(counts[(first[0], first[1])]) / counts[first[0]])
		log_prob += math.log(float(counts[first[0]]) / sum([count for key, count in counts.iteritems() if len(key) == 1]))
	else:
		log_prob += math.log(float(counts[first]) / sum([count for key, count in counts.iteritems() if len(key) == 1]))
		
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
