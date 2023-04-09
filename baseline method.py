#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import sklearn
import json
import pandas as pd
import numpy as np
from collections import Counter
from numpy import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from sklearn.model_selection import GridSearchCV

# # Download files, set up folder, put files into folder

# In[2]:


training_data_path = 'reference_metadata_2013.csv'
test_data_path = 'reference_metadata_2020.csv'

# In[3]:


# specify data type for each column (to be used in pandas read_csv function)
dtype_dict = {'REFERENCE_ID': str, 'TITLE': str, 'AUTHOR': str, 'YEAR': str, 'ABSTRACT': str, 'CITED': int}

# In[4]:


dataframe = pd.read_csv(training_data_path, dtype=dtype_dict, keep_default_na=False)
dataframe

# In[5]:


train_ratio = 0.7  # 70% for training, 30% for validation
random_seed = 100

train_dataframe = dataframe.sample(frac=train_ratio, random_state=random_seed)
valid_dataframe = dataframe.drop(train_dataframe.index)
print('training set size:', len(train_dataframe))
print('validation set size:', len(valid_dataframe))

# In[6]:


test_dataframe = pd.read_csv(test_data_path, dtype=dtype_dict, keep_default_na=False)
test_dataframe


# # Data exploration for training & test data ... YOUR TURN!

# In[7]:


def print_topk_tfidf_words(df, column_name, k):
    counter = Counter()
    for index, row in dataframe.iterrows():
        counter.update(row[column_name].strip().lower().split())

    # sort words by frequency from high to low
    for word, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:k]:
        print(word, count)


# In[8]:


# print_topk_tfidf_words(dataframe, 'TITLE', 100)
# print_topk_tfidf_words(dataframe, 'ABSTRACT', 100)
# print_topk_tfidf_words(test_dataframe, 'TITLE', 100)
# print_topk_tfidf_words(test_dataframe, 'TITLE', 100)


# # Try the trivial baseline: assign random scores to references in the validation set (no learning is needed)

# In[9]:


# Make a list of scores uniformly randomly drawn between 0 and 1 as ranking scores.
# This trivial baseline gives the performance lower bound on the validation set
# Note: we are using average precision as the performance metric
random_pred = [random.random() for i in range(len(valid_dataframe))]
ap = average_precision_score(valid_dataframe['CITED'], random_pred)
print('Average precision of random scoring on validation set:', ap)


# In[10]:


# helper function: write out ranking scores into a csv format file
# params:
#     df: dataframe, where each row is a test example, with column 'REFERENCE_ID' as data id
#     pred: a list or 1-d array of scores for each test example
#     filepath: the output file path
# return:
#     None

def write_test_prediction(df, pred, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write('{},{}\n'.format('REFERENCE_ID', 'Score'))
        for index, row in df.iterrows():
            outfile.write('{},{}\n'.format(row['REFERENCE_ID'], pred[index]))
    print(len(df), 'predictions are written to', filepath)


# In[11]:


random_pred_test = [random.random() for i in range(len(test_dataframe))]
write_test_prediction(test_dataframe, random_pred_test, 'random_score.csv')

# # Build feature extractor

# ## use all unigrams from the 'TITLE' field of training data as features

# In[12]:


vectorizer = CountVectorizer()
vectorizer.fit(train_dataframe['TITLE'])

# # Extract feature vectors for training, validation, and test data

# In[13]:


train_X = vectorizer.transform(train_dataframe['TITLE'])
valid_X = vectorizer.transform(valid_dataframe['TITLE'])
test_X = vectorizer.transform(test_dataframe['TITLE'])
print(train_X.shape)
print(valid_X.shape)
print(test_X.shape)

# # Train model on training set

# In[14]:


# We treat the ranking task as a classification task
# Almost all classification models can output a score that
# indicates (roughly) how confident the model believes
# an example to belong to a class.
# Here in the baseline, we use a logistic regression model.
train_Y = train_dataframe['CITED']
model = LogisticRegression(C=1, solver='liblinear')
model.fit(train_X, train_Y)

# # Evaluate model on training set

# In[15]:


# To produce a ranking score, we ask the model to output
# predicted probability (.predict_proba method), instead of
# predicted class label (.predict method)
train_Y_hat = model.predict_proba(train_X)
train_Y = train_dataframe['CITED'].to_numpy()

# According to the documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
# The predicted probablity for label '1' (CITED) is
# the second column (column index = 1) returned by predict_proba (train_Y_hat)
apt = average_precision_score(train_Y, train_Y_hat[:, 1])
print('Logistic regression, average precision on training set:', apt)

# # Evaluate model on validation set

# In[16]:


valid_Y_hat = model.predict_proba(valid_X)
valid_Y = valid_dataframe['CITED'].to_numpy()
apv = average_precision_score(valid_Y, valid_Y_hat[:, 1])
print('Logistic regression, average precision on validation set:', apv)

# In[17]:


# Here, by using logistic regression, we see a higher average precision
# on the validation set (~0.34) than using random scores (~0.13). It is
# a sanity check confirming that the logistic regression model can learn
# some useful ranking signals (performing better than random).

# Note that in this task, the performance on test data can be lower than
# that on validation set, because the test data and validation set do NOT
# come from the same underlying distribution.
# The validation set is a random subsample of candidate reference pool in 2013.
# The test data is the whole candidate reference pool in 2020, which has a
# different data distribution from 2013 as a result of topic shift in ozone research.


# # After experimentation on the validation set: retrain the final model on all training data, and predict scores for test data

# In[18]:


all_train_Y = dataframe['CITED']
all_train_X = vectorizer.transform(dataframe['TITLE'])
model.fit(all_train_X, all_train_Y)
test_Y_hat = model.predict_proba(test_X)
write_test_prediction(test_dataframe, test_Y_hat[:, 1], 'logistic_regression.csv')

# # Investigate what the model has learned and where it failed (A.K.A. error analysis) ... YOUR TURN!

# param_grid = {
#     'C': np.logspace(-3, 3, 7)
# }
#
# automate = GridSearchCV(model, param_grid, cv=10)
# automate.fit(train_X, train_Y)
# print()
# print("Tuned Hyperparameters :", automate.best_params_)
# print("Accuracy :", automate.best_score_)
