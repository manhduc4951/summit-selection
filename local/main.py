import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import roc_auc_score
import time

# Importing the dataset
dataset = pd.read_csv('path/to/dataset')
X = pd.read_csv('path/to/features/set')
X = X.iloc[:, :].values
y = pd.read_csv('path/to/labels/set')
y = y.iloc[:,0].values

X_train = X
y_train = y

# Feature Scaling --> using for some certain situation
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)

highest_score = 0
optimal = []
avoid_list = []
kf = StratifiedKFold(n_splits=10,random_state=None, shuffle=False)

while True:
    scores = [] # contain scores of each subset
    for feature_index in range(0,X_train.shape[1]):
        if feature_index not in avoid_list:
            X_opt = X_train[:,optimal + [feature_index]]
            # Running each subset "fold" times -> get average score of "fold" times
			# Classifier goes here            
            classifier = DecisionTreeClassifier(random_state = 0)            
            
			# Performance metric selection & Cross validation 	
			cross_scores = cross_val_score(classifier, X_opt, y_train, cv=kf, scoring = 'recall')            
            scores.append(cross_scores.mean())
        else:
            scores.append(0)
    if max(scores) > highest_score:
        highest_score = max(scores)
        optimal.append(scores.index(max(scores)))
        avoid_list.append(scores.index(max(scores)))
    else:
        break 
		
print optimal		