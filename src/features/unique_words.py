from collections import Counter

#####################################
#COUNTING THE NUMBER OF UNIQUE WORDS#
#####################################

def fill_unique_words_column(train_df, valid_df):

    #percentage of unique words to the total number of words
    unique_word_percentages_train = []
    unique_word_percentages_valid = []

    for i in range(len(train_df)):
        splits = train_df.iloc[i]["essay"].split()
        total_words = len(splits)
        unique_words = len(Counter(splits))
        percentage = float(unique_words) / total_words
        unique_word_percentages_train.append(percentage)

    for i in range(len(valid_df)):
        splits = valid_df.iloc[i]["essay"].split()
        total_words = len(splits)
        unique_words = len(Counter(splits))
        percentage = float(unique_words) / total_words
        unique_word_percentages_valid.append(percentage)

    #Add the features to the dataset
    train_df["unique_words"] = unique_word_percentages_train
    valid_df["unique_words"] = unique_word_percentages_valid

    train_df, valid_df = util.append_standardized_column(train_df, valid_df, 'unique_words')

    return train_df, valid_df
