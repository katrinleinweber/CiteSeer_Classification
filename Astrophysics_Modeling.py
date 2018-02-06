#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 02:26:58 2018
Create and Tune Random Forest Model, Draw ROC curve
@author: yiqinshen
"""

# In[Random Forest]:

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(10)


# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to 0.9, then sets the value of that cell as True
# and false otherwise. (10 fold cross validation)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.9

# View the top 5 rows
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

# Create a list of the feature column's NUMBER
features =pd.Index(list(list(df)[i] for i in [106,107,112,121,132,134,135,136,139,140]))


# View features
features

# Create outcome vectors
y = pd.factorize(train['outcome'])[0]

#Random Forest Parameter Setting, Tuned using hpsklearn
clf = RandomForestClassifier(
            bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=0.7717779604354951,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=21,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=70, n_jobs=1, oob_score=False, random_state=4,
            verbose=False, warm_start=False)

# Train the Classifier to take the training features and learn how they relate
# to the training y
clf.fit(train[features], y)

clf.predict(test[features])

clf.predict_proba(test[features])[0:10]

test["outcome"].head()

preds = clf.predict(test[features])

# Crosstab of actuals and predictions

pd.crosstab(test['outcome'], preds, rownames=['Actual'], colnames=['Predict'])

##10 fold cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, df[features], df.outcome, cv=10)

# Print the accuracy across 10 folds
print(scores)

# Print feature importance
list(zip(train[features], clf.feature_importances_))

# In[Tuning random forest parameters based on cross validation]:

from hpsklearn import HyperoptEstimator,any_classifier,random_forest
from hyperopt import tpe

X = df[features].values
y = df["outcome"]

test_size = int( 0.1 * len( y ) )
np.random.seed(10)
indices = np.random.permutation(len(X))
X_train = np.float64(X[ indices[:-test_size]])
y_train = np.float64(y[ indices[:-test_size]])
X_test = np.float64(X[ indices[-test_size:]])
y_test = np.float64(y[ indices[-test_size:]])

estim = HyperoptEstimator(algo=tpe.suggest, trial_timeout=300, classifier = random_forest('my_random_forest') )

estim.fit(X_train, y_train)

print( estim.score( X_test, y_test ) )
# <<show score here>>
print( estim.best_model() )

# In[Another look at error: Out of bag error, plot written by Kian Ho]:

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

print(__doc__)

RANDOM_STATE = 10


# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]


# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore. Change this to the one returned by hpsklearn
min_estimators = 15
max_estimators = 175
   
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(df[features], df.outcome)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

# In[Obtaining probabilities for ROC curve]:

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

X = df[features].values
y = df["outcome"]

kf = model_selection.KFold(n_splits=10)
kf.get_n_splits(y)

#This is the final model returned by hpsklearn
clf = RandomForestClassifier(
        bootstrap=False, class_weight=None, criterion='gini',
        max_depth=None, max_features=0.7717779604354951,
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, min_samples_leaf=21,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        n_estimators=70, n_jobs=1, oob_score=False, random_state=4,
        verbose=False, warm_start=False)
    
column1=[]
column2=[]
for train_index, test_index in kf.split(y):
    X_train = X[list(train_index)]
    y_train = y[list(train_index)]
    X_test = X[list(test_index)]
    y_test = y[list(test_index)]
    clf.fit(X_train, y_train)    
    column1.append(list(y_test))
    column2.append(list(clf.predict_proba(X_test)[:,1]))
    
def flatList(x):
    flat = [item for sublist in x for item in sublist]
    return flat


column1=flatList(column1)
column2=flatList(column2)

roc_auc_score(column1, column2)


# In[Drawing a ROC curve]:

# Import necessary modules
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(column1, column2)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
