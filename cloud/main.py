import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import roc_auc_score
import time


def parseFunc(X):
	# Classifier goes here
    classifier = DecisionTreeClassifier(random_state = 0)

	# Performance metric selection & Cross validation 		
    cross_scores = cross_val_score(classifier, X[0], y_train_opt, cv=kf, scoring = 'recall')
    return (X[1],cross_scores.mean())


def findTheKey(lst):    
    temp_d = []
    for i in lst:
        temp_d.append(i[1])
    for i in lst:
        if i[1] == max(temp_d):
            return i[0]



def findMax(lst):   
    temp_d = []
    for i in lst:
        temp_d.append(i[1])
    return max(temp_d)



dataset = pd.read_csv('path/to/dataset')
X = pd.read_csv('path/to/features/set')
X = X.iloc[:, :].values
y = pd.read_csv('path/to/labels/set')
y = y.iloc[:,0].values

X_train_opt = X[:]
y_train_opt = y[:]

# Feature Scaling --> using for some certain situation
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train_opt = sc_X.fit_transform(X_train_opt)

highest_score = 0
optimal = []
avoid_list = []
kf = StratifiedKFold(n_splits=10,random_state=None, shuffle=False)

while True:
    data_train_list = []
    for feature_index in range(0, X_train_opt.shape[1]):
        if feature_index not in avoid_list:
            X_opt = X_train_opt[:,optimal + [feature_index]]
            data_train_list.append((X_opt,feature_index))
    data_train_list = sc.parallelize(data_train_list,len(data_train_list))
    lst = data_train_list.map(lambda X: parseFunc(X) )
    lst = lst.collect()
    maxVal = findMax(lst)
    if maxVal > highest_score:
        highest_score = maxVal
        maxKey = findTheKey(lst)
        optimal.append(maxKey)
        avoid_list.append(maxKey)
    else:
        break

print optimal		

