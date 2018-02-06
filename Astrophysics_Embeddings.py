#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:23:29 2018
Turning Tokenized Fulltext into Embeddings
@author: yiqinshen
"""

# In[Adding Term frequency inverse document frequency (TF-IDF)]:

###########
###########

%matplotlib inline
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import operator
import re, os
import sys
import nltk
from pprint import *
from sklearn.manifold import TSNE
import multiprocessing
import codecs
import gensim.models.word2vec as w2v
from nltk.corpus import stopwords
from nltk import FreqDist
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle
import logging

# In[Cleaning Text]:

from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
from nltk.corpus import stopwords
import gensim 
import pandas as pd

#Read in document with tokenized words
df = pd.read_csv("/Users/yiqinshen/Dropbox/Insight/merge_by_primary_full.csv",encoding="latin1")

# Count Vectorizer
count_vect = CountVectorizer()
data_matrix_CV = count_vect.fit_transform(df['citation_text_clean'].values.astype('U'))
tfidf_transformer = TfidfTransformer()
datanodata_matrix_tfidf = tfidf_transformer.fit_transform(data_matrix_CV)
targets = df['outcome'].values
data_features  = count_vect.get_feature_names()


# In[Visualiziing Top Features in TFIDF]:

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    
    
df_class_list = top_feats_by_class(datanodata_matrix_tfidf, targets, data_features)

plot_tfidf_classfeats_h(df_class_list)

# In[Using TF-IDF Dimension Reduction]:

from sklearn import manifold

# Do the dimension reduction
#
k = 10 # number of nearest neighbors to consider
d = 2 # dimensionality
pos = manifold.Isomap(k, d, eigen_solver='auto').fit_transform(datanodata_matrix_tfidf.toarray())

#Add dimension reduction to data frame
df = pd.concat([df,pos])


# Visualizing meaningful "cluster" labels
# Apply a word label if the clusters max TF-IDF is in the 99% quantile of the whole corpus of TF-IDF scores
labels = count_vect.get_feature_names() #text labels of features
clusterLabels = []
t99 = scipy.stats.mstats.mquantiles(datanodata_matrix_tfidf.toarray(), [ 0.99])[0]

for i in range(0,datanodata_matrix_tfidf.shape[0]):
    row = datanodata_matrix_tfidf.getrow(i)
    if row.max() >= t99:
        arrayIndex = np.where(row.data == row.max())[0][0]
        clusterLabels.append(labels[row.indices[arrayIndex]])
    else:
        clusterLabels.append('')

# Plot the dimension reduced data

matplotlib.pyplot.xlabel('reduced dimension-1')
matplotlib.pyplot.ylabel('reduced dimension-2')
for i in range(0,62):
    matplotlib.pyplot.scatter(pos[i][0], pos[i][1], c='cyan')
    matplotlib.pyplot.annotate(clusterLabels[i], xy=(pos[i][0], pos[i][1]), xytext=None, xycoords='data', textcoords='data', arrowprops=None)

matplotlib.pyplot.show()
