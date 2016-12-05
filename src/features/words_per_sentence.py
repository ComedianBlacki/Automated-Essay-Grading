# IMPORT NECESSARY LIBRARIES
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import discriminant_analysis as da
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score
from nltk.corpus import wordnet

################################################################
#######################  FROM MILESTONE 4 ######################  
################################################################

def append_regularized_scores(old_df):
    new_df = old_df.copy()
    new_df['std_score'] = new_df.groupby(['essay_set'])[['score']].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    return new_df

def create_regularization_data(old_df):
    #getting the number of datasets
    max_essay_set = max(old_df['essay_set'])
    #list of the regularized values
    regularization_data = []
    for i in range(max_essay_set+1):
        mean = np.mean((old_df[old_df['essay_set'] == i + 1])['score'])
        std = np.std((old_df[old_df['essay_set'] == i + 1])['score'])
        regularization_data.append([i + 1, mean, std])
    return regularization_data

train_cols = ['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score']
train_df = pd.read_csv('../../data/training_set_rel3.tsv', delimiter='\t', usecols=train_cols)
for i in xrange(train_df.shape[0]):
    if not np.isnan(train_df.get_value(i, 'domain2_score')):
        assert train_df.get_value(i, 'essay_set') == 2
        new_val = train_df.get_value(i, 'domain1_score') + train_df.get_value(i, 'domain2_score')
        train_df.set_value(i, 'domain1_score', new_val) 
train_df = train_df.drop('domain2_score', axis=1)
train_df = train_df.rename(columns={'domain1_score': 'score'})

regularization_data = create_regularization_data(train_df)
train_df = append_regularized_scores(train_df)
# Read in validation data
valid_cols = ['essay_id', 'essay_set', 'essay', 'domain1_predictionid', 'domain2_predictionid']
valid_df = pd.read_csv('../../data/valid_set.tsv', delimiter='\t', usecols=valid_cols)
valid_df['score'] = pd.Series([0] * valid_df.shape[0], index=valid_df.index)

# scores are stored in separate data set, we'll put them in same one
valid_scores = pd.read_csv('../../data/valid_sample_submission_5_column.csv', delimiter=',')
# put each score in our data set, and make sure to handle essay set 2
for i in xrange(valid_df.shape[0]):
    dom1_predid = valid_df.get_value(i, 'domain1_predictionid')
    row = valid_scores[valid_scores['prediction_id'] == dom1_predid]
    score = row.get_value(row.index[0], 'predicted_score')
    dom2_predid = valid_df.get_value(i, 'domain2_predictionid')
    if not np.isnan(dom2_predid):
        assert valid_df.get_value(i, 'essay_set') == 2
        rowB = valid_scores[valid_scores['prediction_id'] == dom2_predid]
        scoreB = rowB.get_value(rowB.index[0], 'predicted_score')
        score += scoreB  
    valid_df.set_value(i, 'score', score)
valid_df = valid_df.drop(['domain1_predictionid', 'domain2_predictionid'], axis=1)

# returned a copy of old_df, with essays cleaned for count vectorizer
# cleaning returns essay with only lowercase words separated by space
def vectorizer_clean(old_df):
    new_df = old_df.copy()
    for i in xrange(new_df.shape[0]):
        new_df.set_value(i, 'essay', " ".join(re.sub('[^a-zA-Z\d\s]', '', new_df['essay'].iloc[i]).lower().split())) 
    return new_df

# print essays cleaned for vectorizer (essay is now just lowercase words separated by space) 
vectorizer_train = vectorizer_clean(train_df)
vectorizer_valid = vectorizer_clean(valid_df)
vectorizer = TfidfVectorizer(stop_words = 'english')
train_essays = vectorizer_train['essay'].values
train_vectors = vectorizer.fit_transform(train_essays).toarray()
#normalizing for y
train_std_scores = np.asarray(vectorizer_train['std_score'], dtype="|S6")

##################################################################
#################### NUM OF SENTENCES FEATURE ####################
##################################################################

# REGULARIZING FOR SENTENCE LENGTHS 
def append_regularized_sentence_length(old_df):
    new_df = old_df.copy()
    new_df['std_sentence_len'] = new_df.groupby(['essay_set'])[['sentence_length']].apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    return new_df

def create_regularization_sentence_length(old_df):
    #getting the number of datasets
    max_essay_set = max(old_df['essay_set'])
    #list of the regularized values
    regularization_data = []
    for i in range(max_essay_set+1):
        mean = np.mean((old_df[old_df['essay_set'] == i + 1])['sentence_length'])
        std = np.std((old_df[old_df['essay_set'] == i + 1])['sentence_length'])
        regularization_data.append([i + 1, mean, std])
    return regularization_data

def sentences(par):
    split_sent = re.split(r'[.!?]+', par)
    return len(split_sent)

numOfSent_train = []
for essay in train_df['essay']:
    sent = sentences(essay)
    numOfSent_train.append(sent)

numOfSent_valid = []
for essay in valid_df['essay']:
    sent = sentences(essay)
    numOfSent_valid.append(sent)

train_df['sentence_length'] = numOfSent_train
valid_df['sentence_length'] = numOfSent_valid

regularization_data_sentence = create_regularization_sentence_length(train_df)
train_df = append_regularized_sentence_length(train_df)

# FITTING THE TRAINING SET USING L2 LOGISTIC
logistic_l2 = LogReg(penalty='l2', solver='liblinear', n_jobs=4)
xs = [[x] for x in np.array(train_df['sentence_length'])]
logistic_l2.fit(xs, train_std_scores)

# DENORMALIZING FOR THE VALID SET
max_essay_set = max(train_df['essay_set'])
stand_pred_values_l2 = []
for i in range(max_essay_set):
    current_set = valid_df[valid_df['essay_set'] == i + 1]['sentence_length']
    for value in current_set:
        stand_pred_values_l2.append(int(float(value) * float(regularization_data_sentence[i][2]) + (regularization_data_sentence[i][1])))

# PREDICTING THE SCORE USING THE NEW SENTENCE LENGTH
valid_df['new_sentence_length_std'] = stand_pred_values_l2
valid_x = [[x] for x in np.array(valid_df['new_sentence_length_std'])]
valid_pred_std_scores_l2 = logistic_l2.predict(valid_x)
valid_df["Log_L2 predicted_scores"] = valid_pred_std_scores_l2

# denormalizing the values and placing them into the stand_pred_values array
stand_pred_values_l2 = []
for i in range(max_essay_set):
    current_set = valid_df[valid_df['essay_set'] == i + 1]['Log_L2 predicted_scores']
    for value in current_set:
        stand_pred_values_l2.append(int(float(value) * float(regularization_data[i][2]) + (regularization_data[i][1])))

valid_df['newly_predicted_scores_log_l2'] = stand_pred_values_l2

# FITTING THE TRAINING SET USING L2 LOGISTIC
logistic_l1 = LogReg(penalty='l1', solver='liblinear', n_jobs=4)
logistic_l1.fit(xs, train_std_scores)

valid_pred_std_scores_l1 = logistic_l1.predict(valid_x)
valid_df["Log_L1 predicted_scores"] = valid_pred_std_scores_l1
stand_pred_values_l1 = []
for i in range(max_essay_set):
    current_set = valid_df[valid_df['essay_set'] == i + 1]['Log_L1 predicted_scores']
    for value in current_set:
        stand_pred_values_l1.append(int(float(value) * float(regularization_data[i][2]) + (regularization_data[i][1])))
valid_df['newly_predicted_scores_log_l1'] = stand_pred_values_l1


###############
#   Scoring   #
###############

#Scoring the predicted values with the actual values
log_l1_count = 0
log_l2_count = 0
for i in range(len(valid_df)):
    if valid_df.iloc[i]['score'] == valid_df.iloc[i]['newly_predicted_scores_log_l2']:
        log_l2_count += 1
    if valid_df.iloc[i]['score'] == valid_df.iloc[i]['newly_predicted_scores_log_l1']:
        log_l1_count += 1
        
print "LOGISTIC L2 using Feature: Number of Sentences"
print "Number of correct predictions =", log_l2_count
print "Total number of observations =", len(valid_df)
print "Score =", float(log_l2_count) / len(valid_df)

print ""
print "LOGISTIC L1 using Feature: Number of Sentences"
print "Number of correct predictions =", log_l1_count
print "Total number of observations =", len(valid_df)
print "Score =", float(log_l1_count) / len(valid_df)



