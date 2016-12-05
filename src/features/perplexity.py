# doesn't work because current nltk version doesn't have model package
from nltk.model import LaplaceNgramModel
from nltk.tokenize import word_tokenize

def ngram_counter(train_essays):
	

# returns an nltk NgramModel fit on the training essays, for which
# eperplexity can then be calculated using the perplexity function
# essay should be cleaned via milestone 4 cleaning before being passed in
def ngram_model(train_essays):

	return LaplaceNgramModel(train_essays)

# After having already fit model on a set of training essays, calculates the
# perplexity of a student's essay based from the model, and returns this
# perplexity to be used as a feature
def perplexity(test_essay, model):
	return model.perplexity(test_essay)