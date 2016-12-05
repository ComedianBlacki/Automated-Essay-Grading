import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score

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

# Read in training data
# Note that for essay set 2, score becomes average of 2 domain scores
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

print "The regularized data for each essay set = ", regularization_data
print "\n"

#print train_df[train_df['essay_set'] == 2].head()
print "Head of Training Data Frame"
print train_df.head()
print "\n"

#validate that the standardization works
max_essay_set = max(train_df['essay_set'])
for i in range (max_essay_set):
    valid = train_df[train_df["essay_set"] == i + 1]["std_score"]
    std = np.std(valid)
    mean = np.mean(valid)
    if mean < 0.000001:
        mean = 0
    print "Mean and Std Dev of essay set " + str(i + 1) + ": ", mean, ",", std

# Show nothing is empty in training set
print "\n"
if train_df.isnull().any().any():
    print 'Training data is missing!'
else:
    print 'No missing training data!'

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
#print valid_df[valid_df['essay_set'] == 2].head()
print "\n"
print "Head of Validation Data Frame"
print valid_df.head()

# Show nothing is empty in validation set
print "\n"
if valid_df.isnull().any().any():
    print 'Validation data is missing!'
else:
    print 'No missing validation data!'

# returned a copy of old_df, with essays cleaned for count vectorizer
# cleaning returns essay with only lowercase words separated by space
def vectorizer_clean(old_df):
    new_df = old_df.copy()
    for i in xrange(new_df.shape[0]):
        new_df.set_value(i, 'essay', " ".join(re.sub('[^a-zA-Z\d\s]', '', new_df['essay'].iloc[i]).lower().split())) 
    return new_df

# print essays cleaned for vectorizer (essay is now just lowercase words separated by space) 
vectorizer_train = vectorizer_clean(train_df)
print vectorizer_train.head()

# print essays cleaned for vectorizer (essay is now just lowercase words separated by space) 
vectorizer_valid = vectorizer_clean(valid_df)
print vectorizer_valid.head()

vectorizer = TfidfVectorizer(stop_words = 'english')

# vectorizer2 = TfidfVectorizer(stop_words = 'english', ngram_range=(2,2))
# vectorizer3 = TfidfVectorizer(stop_words = 'english', ngram_range=(3,3))
# vectorizer4 = TfidfVectorizer(stop_words = 'english', ngram_range=(4,4))
# vectorizer5 = TfidfVectorizer(stop_words = 'english', ngram_range=(5,5))


#Get all the text from data
train_essays = vectorizer_train['essay'].values

#Turn each text into an array of word counts
train_vectors = vectorizer.fit_transform(train_essays).toarray()

# train_vectors2 = vectorizer2.fit_transform(train_essays).toarray()
# train_vectors3 = vectorizer3.fit_transform(train_essays).toarray()
# train_vectors4 = vectorizer4.fit_transform(train_essays).toarray()
# train_vectors5 = vectorizer5.fit_transform(train_essays).toarray()


#normalizing for y
train_std_scores = np.asarray(vectorizer_train['std_score'], dtype="|S6")
print train_std_scores[:5]

######################################
## TfidfVectorizer with ngram=(1,1) ##
######################################


###############
# Logistic L2 #
###############

# Logistic Model with L2 penalty
logistic_l2 = LogReg(penalty='l2', solver='liblinear', n_jobs=4)
logistic_l2.fit(train_vectors, train_std_scores)

valid_vectors = vectorizer.transform(vectorizer_valid['essay'].values).toarray()

# My guess is we will want to denormalize these scores for quadratic weighted k
valid_pred_std_scores_l2 = logistic_l2.predict(valid_vectors)

# Appending predicted scores to validation data set
valid_df["Log_L2 predicted_scores"] = valid_pred_std_scores_l2

#denormalizing the values and placing them into the stand_pred_values array
stand_pred_values_l2 = []
for i in range(max_essay_set):
    current_set = valid_df[valid_df['essay_set'] == i + 1]['Log_L2 predicted_scores']
    for value in current_set:
        stand_pred_values_l2.append(int(float(value) * float(regularization_data[i][2]) + (regularization_data[i][1])))
# print stand_pred_values_l2

#adding the denormalizede predicted values to the valid_df dataset
valid_df['newly_predicted_scores_log_l2'] = stand_pred_values_l2

###############
# Logistic L1 #
###############

# Logistic Model with L1 penalty
logistic_l1 = LogReg(penalty='l1', solver='liblinear', n_jobs=4)
logistic_l1.fit(train_vectors, train_std_scores)

valid_pred_std_scores_l1 = logistic_l1.predict(valid_vectors)


# Appending predicted scores to validation data set
valid_df['Log_L1 predicted_scores'] = valid_pred_std_scores_l1

#denormalizing the values and placing them into the stand_pred_values array
stand_pred_values_l1 = []
for i in range(max_essay_set):
    current_set = valid_df[valid_df['essay_set'] == i + 1]['Log_L1 predicted_scores']
    for value in current_set:
        stand_pred_values_l1.append(int(float(value) * float(regularization_data[i][2]) + (regularization_data[i][1])))
# print stand_pred_values_l1

#adding the denormalizede predicted values to the valid_df dataset
valid_df['newly_predicted_scores_log_l1'] = stand_pred_values_l1

###############
#   Scoring   #
###############

#Scoring the predicted values with the actual values
log_l2_count = 0
log_l1_count = 0
for i in range(len(valid_df)):
    if valid_df.iloc[i]['score'] == valid_df.iloc[i]['newly_predicted_scores_log_l2']:
        log_l2_count += 1
    if valid_df.iloc[i]['score'] == valid_df.iloc[i]['newly_predicted_scores_log_l1']:
        log_l1_count += 1
        
print "LOGISTIC L2"
print "Number of correct predictions =", log_l2_count
print "Total number of observations =", len(valid_df)
print "Score =", float(log_l2_count) / len(valid_df)

print ""
print "LOGISTIC L1"
print "Number of correct predictions =", log_l1_count
print "Total number of observations =", len(valid_df)
print "Score =", float(log_l1_count) / len(valid_df)

#Spearman Correlation Coefficient
from scipy.stats import spearmanr as Spearman

print "Logistic L2:", Spearman(a = valid_df["score"], b = valid_df["newly_predicted_scores_log_l2"])
print "Logistic L1:", Spearman(a = valid_df["score"], b = valid_df["newly_predicted_scores_log_l1"])