
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from imblearn.combine import SMOTEENN

def visualizeAll(X):
    X = X.drop(["y"], axis = 1)
    plt.hist(X["balance"])
    plt.xlabel("balance")
    plt.ylabel("Frequency")
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.tight_layout()


def loadDataSet(dataset):
    originaldf = pd.read_csv(dataset, sep = ";")
    ##visualizeAll(originaldf)
    objectdf = originaldf.drop(["age","balance","day","duration","campaign","pdays","previous", "y"], axis = 1)
    integerdf = originaldf.drop(["job", "marital", "education", "default",
    "housing","loan","contact", "month", "poutcome", "y"], axis = 1)
    dataset = originaldf.values
    data = dataset[:,:-1]
    target = dataset[:,-1]
    return data, target, originaldf, objectdf, integerdf

def encodedInputs(arr):
    oe = OrdinalEncoder()
    oe.fit(arr)
    objectsEncoded = oe.transform(arr)
    return objectsEncoded

def encodedTargets(target):
    le = LabelEncoder()
    le.fit(target)
    targetsEncoded = le.transform(target)
    return targetsEncoded

def selectFeatures(X_train, X_test, y_train):
    fs = SelectKBest(score_func=mutual_info_classif, k = "all")
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def imblanaceCorrection(X,y):
    oversample = SMOTEENN( sampling_strategy = 0.6)
    X,y = oversample.fit_resample(X,y)
    return X,y

def featureScoreBar(scores):
    for i in range(len(scores)):
        print("Feature " + str(i) + " :" + str(scores[i]))
    sns.barplot([i for  i in range(len(scores))], scores)

def alphaPruning(ccp_alphas): ## Used to find the minimum ccp_aplha for tree pruning, which is 5.083e-5

    clfs = []

    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        print(clf)
    return clfs

def depthsVsAlphas(clfs, ccp_alphas):
    tree_depths = [clf.tree_.max_depth for clf in clfs]
    plt.figure(figsize = (10,6))
    plt.plot(ccp_alphas[:-1], tree_depths[:-1])

def scoresVsAlphas(clfs, ccp_alphas):
    acc_scores = [metrics.accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]

    plt.figure(figsize = (10,6))
    plt.plot(ccp_alphas[:-1], acc_scores[:-1])


def stdDeviation(A):
    standard = np.std(A)
    return standard

def DTC(score_handler):
    clf = DecisionTreeClassifier(criterion = "gini", ccp_alpha = 6.64749e-5, max_depth = 20)
    skf = StratifiedKFold(n_splits=5, random_state = 1, shuffle = True)
    skf.get_n_splits(X,y)
    for train_index, test_index in skf.split(X,y):

        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("F1 Score:", metrics.f1_score(y_test, y_pred))

clf = MLPClassifier(random_state = 1, max_iter = 1000, activation = "logistic")
skf = StratifiedKFold(n_splits=5, random_state = 1, shuffle = True)
skf.get_n_splits(X,y)
for train_index, test_index in skf.split(X,y):

    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))



X, y, df, objectdf, integerdf = loadDataSet("bank-full.csv")

pd.set_option("display.max_columns", None)
objectdfArr = objectdf.to_numpy()
X = pd.DataFrame(encodedInputs(objectdfArr), columns = objectdf.columns)
X = X.join(integerdf)
y = encodedTargets(y)
target = pd.DataFrame(y, columns = ["y/n"]) ##original target class labels
lengthofTarget = len(target) ## the length of the feature columns

X,y = imblanaceCorrection(X,y)
X = X.drop(["default", "age"], axis = 1)
print(X.shape)

plt.show()
