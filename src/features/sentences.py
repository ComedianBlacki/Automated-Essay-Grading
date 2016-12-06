# IMPORT NECESSARY LIBRARIES
import re
import util

def sentences(par):
    split_sent = re.split(r'[.!?]+', par)
    return len(split_sent)

def fill_sentence_column(train_df, valid_df):
    numOfSent_train = []
    for essay in train_df['essay']:
        sent = sentences(essay)
        numOfSent_train.append(sent)

    numOfSent_valid = []
    for essay in valid_df['essay']:
        sent = sentences(essay)
        numOfSent_valid.append(sent)

    train_df['sentence_count'] = numOfSent_train
    valid_df['sentence_count'] = numOfSent_valid

    train_df, valid_df = util.append_standardized_column(train_df, valid_df, 'sentence_count')

    return train_df, valid_df
