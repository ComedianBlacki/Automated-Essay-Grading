import math
import util

def fill_perplexity_column(train_df, valid_df):
	train_clean = util.vectorizer_clean(train_df)
	valid_clean = util.vectorizer_clean(valid_df)

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
	return train_df, valid_df

class LaPlaceCounter(dict):
	def __getitem__(self, idx):
		self.setdefault(idx, 1)
		return dict.__getitem__(self, idx)

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# train_essays must
def create_counts(train_essays):
	n_gram_counts = LaPlaceCounter()
	for essay in train_essays:
		word_list = essay.split()
		for i in xrange(3):
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
		log_prob += math.log(float(counts[tri_gram]) / counts[(tri_gram[0], tri_gram[1])])

	# handle when essays are shorter than 3 words
	last = None
	if len(tri_grams) > 0:
		last = tri_grams[len(tri_grams) - 1]
	elif len(find_ngrams(word_list, 2)) > 0:
		bi_grams = find_ngrams(word_list, 2)
		last = bi_grams[len(bi_grams) - 1]
	else:
		uni_grams = find_ngrams(word_list, 1)
		last = uni_grams[len(uni_grams) - 1]

	if len(last) > 1:
		log_prob += math.log(float(counts[(last[0], last[1])]) / counts[last[0]])
		log_prob += math.log(float(counts[last[0]]) / sum(counts.values()))
	else:
		log_prob += math.log(float(counts[last]) / sum(counts.values()))
		
	return math.exp(-log_prob / len(test_essay))

def main():
	train_essays = [
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin",
	"hi my name is kevin"
	]

	test_essay = "what are your favorite color"
	test_essay2 = "my name is annie kevin"
	test_essay3 = "hi my name is kevin"

	counts = create_counts(train_essays)
	print perplexity(counts, test_essay)
	print perplexity(counts, test_essay2)
	print perplexity(counts, test_essay3)

	print counts

if __name__ == "__main__": main()
