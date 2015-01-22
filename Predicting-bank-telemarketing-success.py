import pandas as pd
from sklearn import preprocessing
import argparse
import numpy as np
import scipy
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from time import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.grid_search import ParameterGrid
import pylab as pl
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from scipy import interp
from sklearn.cross_validation import StratifiedKFold

###############################
data = pd.read_csv('bank-additional-full.csv', delimiter=';')

# Label Encoding
lab = preprocessing.LabelEncoder()
data.job = lab.fit_transform(data.job)
data.marital = lab.fit_transform(data.marital)
data.education = lab.fit_transform(data.education)
data.default = lab.fit_transform(data.default)
data.housing = lab.fit_transform(data.housing)
data.loan = lab.fit_transform(data.loan)
data.contact = lab.fit_transform(data.contact)
data.month = lab.fit_transform(data.month)
data['day_of_week'] = lab.fit_transform(data['day_of_week'])
data.poutcome = lab.fit_transform(data.poutcome)
data.y = lab.fit_transform(data.y)

features = data.as_matrix()
print type(features)
print features.shape
target = features[:,20]
features_numeric = features[0:41188,0:20]

# One-k-scheme encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
cat_indices = np.array([[1,2,3,4,5,6,7,8,9,14]])

enc = OneHotEncoder(categorical_features = cat_indices)
encoded_features1 = enc.fit_transform(features_numeric)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(encoded_features1.todense(), target, test_size=0.4, random_state=0)

parser = argparse.ArgumentParser(description='Predicting bank telemarketing success')
parser.add_argument('-id', type=int, choices=[1,2,3,4])
args = parser.parse_args()

classifiers = (None, RandomForestClassifier, SVC, LogisticRegression, DecisionTreeClassifier)

grid1 = [{'n_estimators' : [50, 100, 250, 500, 1000]}]
grid2 = [{'kernel': ['rbf', 'linear'], 'C': [0.1, 1.0, 100.0],'gamma': [0.001, 0.1, 10.0], 'class_weight' : ['auto'], 'probability' : [True]}]
grid3 = [{'C' : [1], 'class_weight' : ['auto']}]
grid4 = [{'max_features' : ['sqrt', 'log2', 20]}]

grids = (None, grid1, grid2, grid3, grid4)
grid_obj = grids[args.id]
cls_obj = classifiers[args.id]

best_score = None
print ("Performing cross validation using parameter grid...")
for one_param in ParameterGrid(grid_obj):
    cls = cls_obj(**one_param)
    
    one_score = cross_val_score(cls, X_train, Y_train, cv=5, scoring = 'roc_auc')
    mscore = one_score.mean()
   
    print ("param=%s, score=%.6f" % (repr(one_param),mscore))
            
    if ( best_score is None or best_score < mscore): 
        best_param = one_param
        best_score = mscore
        best_svc = cls

print ("Best score for Cross Validation: %.6f" % best_score)

### ROC
roc_name = (None, "Random Forest", "SVM", "Logistic Regression", "Decision Trees")
probas_ = best_svc.fit(X_train, Y_train).predict_proba(X_test)
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic- %s' % (roc_name[args.id]))
pl.legend(loc="lower right")
pl.show()