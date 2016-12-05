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