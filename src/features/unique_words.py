from collections import Counter
import util

#####################################
#COUNTING THE NUMBER OF UNIQUE WORDS#
#####################################

def fill_unique_words_column(train_df, valid_df, train_essays, valid_essays):

    #percentage of unique words to the total number of words
    unique_word_percentages_train = []
    unique_word_percentages_valid = []

    for i in range(len(train_essays)):
        splits = train_essays[i].split()
        unique_words = len(Counter(splits))
        unique_word_percentages_train.append(unique_words)

    for i in range(len(valid_essays)):
        splits = valid_essays[i].split()
        unique_words = len(Counter(splits))
        unique_word_percentages_valid.append(unique_words)

    #Add the features to the dataset
    train_df["unique_words"] = unique_word_percentages_train
    valid_df["unique_words"] = unique_word_percentages_valid

    train_df, valid_df = util.append_standardized_column(train_df, valid_df, 'unique_words')

    return train_df, valid_df
