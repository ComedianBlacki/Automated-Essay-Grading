from nltk.corpus import wordnet

# input is list of words in text, output proportion spelling correct
def proportion_correct_spelling(text):
    text_len = len(text)
    correct = 0
    for word in text:
        try:
            if wordnet.synsets(word):
                correct += 1
        except:
            correct+= 0
    return 1. * correct / text_len

# should be the cleaned up version 

def fill_spelling_column(train_df, valid_df, train_essays, valid_essays):

	spelling_feature_x = []
	for i, train in enumerate(train_essays):
		if i % 20 == 0:
			print "Train essay " + str(i) + " of " + str(train_essays.shape[0])
		sentence = train.split()
		percent = proportion_correct_spelling(sentence)
		spelling_feature_x.append(percent)

	valid_spelling_x = []
	for i, valid in enumerate(valid_essays):
		if i % 20 == 0:
			print "Valid essay " + str(i) + " of " + str(valid_essays.shape[0])
		sentence = valid.split()
		percent = proportion_correct_spelling(sentence)
		valid_spelling_x.append(percent)

	train_df['spelling_correct'] = spelling_feature_x
	valid_df['spelling_correct'] = valid_spelling_x
	return train_df, valid_df


