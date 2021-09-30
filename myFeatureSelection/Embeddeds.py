from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd

def SVR_RFE(x,y,nfSelected,FId):
    
    bestFId = []
    estmr = SVR(kernel='linear', C=1e3)

    selector = RFE(estimator = estmr, n_features_to_select = nfSelected) 
    selector = selector.fit(x, y)

    #here we obtain index of genes that have rank = 1 in RFE process:
    indexRankedF = []
    index = 0
    for item in selector.ranking_:
        if item == 1:
            indexRankedF.append(index)
        index+=1

    RankedF = []
    for ind in indexRankedF:
        RankedF.append(FId[ind])
    
    return RankedF

def FIGBR(x,y,nfSelected,col):
    x_new= x.iloc[0:,col]
    clf = GradientBoostingRegressor(n_estimators=50)
    clf = clf.fit(x_new, y)
    scores = clf.feature_importances_
    scores = pd.Series(scores)
    ind = scores.nlargest(nfSelected).index
    listOfSF = list()
    for i in ind:
        listOfSF.append(i)
    listOfSF.sort()
    return listOfSF

def FIAdaBoost(x,y,nfSelected,col):
    x_new= x.iloc[0:,col]
    clf = AdaBoostRegressor(n_estimators=50)
    clf = clf.fit(x_new, y)
    scores = clf.feature_importances_
    scores = pd.Series(scores)
    ind = scores.nlargest(nfSelected).index
    listOfSF = list()
    for i in ind:
        listOfSF.append(i)
    listOfSF.sort()
    return listOfSF

def FIExtra(x,y,nfSelected,col):
    x_new= x.iloc[0:,col]
    clf = ExtraTreesRegressor(n_estimators=50)
    clf = clf.fit(x_new, y)
    scores = clf.feature_importances_
    scores = pd.Series(scores)
    ind = scores.nlargest(nfSelected).index
    listOfSF = list()
    for i in ind:
        listOfSF.append(i)
    listOfSF.sort()
    return listOfSF

def FIRF(x,y,nfSelected,col):
    x_new= x.iloc[0:,col]
    clf = RandomForestRegressor(n_estimators=50)
    clf = clf.fit(x_new, y)
    scores = clf.feature_importances_
    scores = pd.Series(scores)
    ind = scores.nlargest(nfSelected).index
    listOfSF = list()
    for i in ind:
        listOfSF.append(i)
    listOfSF.sort()
    return listOfSF