import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import util

UNIV_TAGS = set(['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X'])

# essay should NOT be cleaned
# returns proportion of each part of speech in essay
def create_tags_dict(essay):
	text = word_tokenize(essay)
	num_tokens = len(text)
	tagged_words = nltk.pos_tag(text, tagset='universal')
	tags_only = [tag for _, tag in tagged_words]
	fd = FreqDist(tags_only)
	tags_dict = {}
	for pos in UNIV_TAGS:
		tags_dict[pos] = float(fd[pos]) / num_tokens

	return tags_dict

def fill_pos_columns(train_df, valid_df):

	# Add parts of speech columns
	train_df = util.append_zeros_column(train_df, 'noun')
	train_df = util.append_zeros_column(train_df, 'verb')
	train_df = util.append_zeros_column(train_df, 'adj')

	valid_df = util.append_zeros_column(valid_df, 'noun')
	valid_df = util.append_zeros_column(valid_df, 'verb')
	valid_df = util.append_zeros_column(valid_df, 'adj')

	dfs = [train_df, valid_df]

	for j, df in enumerate(dfs):
		for i in xrange(df.shape[0]):
			if i % 1000 == 0:
				essay_set = None
				if j == 0:
					essay_set = "Train"
				else:
					essay_set = "Validation"
				print essay_set + " essay " + str(i) + " of " + str(df.shape[0])
			essay = df.get_value(i, 'essay').decode('utf-8',errors='ignore')
			tags = create_tags_dict(essay)

			#print tags

			df = df.set_value(i, 'noun', tags['NOUN'])
			df = df.set_value(i, 'verb', tags['VERB'])
			df = df.set_value(i, 'adj', tags['ADJ'])

	return dfs[0], dfs[1]