import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from skrebate import ReliefF

def FRegression(x,y,numOfSL,col):
    x_new= x.iloc[0:,col]
    y_new = list()
    for j in y:
        y_new.append(float(j))
    listOfSF = SelectKBest(f_regression, k=numOfSL).fit(x_new, y_new).get_support(True)
    return listOfSF


def MutualInfoRegression(x,y,numOfSL,col):
    x_new= x.iloc[0:,col]
    y_new = list()
    for j in y:
        y_new.append(float(j))
    listOfSF = SelectKBest(mutual_info_regression, k=numOfSL).fit(x_new, y_new).get_support(True)
    return listOfSF

# def pearsonloop(x,y):
#     corr , p_val = pearsonr(x, y)
#     return abs(corr)

# def myPearson(x,y,NumOfSlc,col):
    
#     xFrame = pd.DataFrame(x)
#     pearsonVal = []
#     pearsonVal = Parallel(n_jobs = 6)(delayed(pearsonloop)(xFrame[i], y) for i in col)
   
#     np_pearsonVal = np.array(pearsonVal)
#     ind = np.argsort(np_pearsonVal)[-NumOfSlc:][::-1]   
#     val = np_pearsonVal[ind]
    
#     return (ind,val)

def myPearson(x,y,NumOfSlc,col):
    
    #xFrame = pd.DataFrame(x)
    pearsonVal = []
    
    for i in col:
      corr , p_val = pearsonr(x[i], y)
      pearsonVal.append(abs(corr))

    np_pearsonVal = np.array(pearsonVal)
    ind = np.argsort(np_pearsonVal)[-NumOfSlc:][::-1]   
    val = np_pearsonVal[ind]
    
    return (ind,val)


def varF(x,NumOfSlc,col):
    x_new= x.iloc[0:,col]
    scores=x_new.var(0)
    ind = scores.nlargest(NumOfSlc).index
    listOfSF = list()
    for i in ind:
        listOfSF.append(i)
    listOfSF.sort()
    return listOfSF


def relief(x,y,NumOfSL,col):
    x_new= x.iloc[0:,col]

    relf = ReliefF(n_neighbors=10 , n_jobs = 6)
    relf.fit(x_new.values, y.values)

    indOfSF = relf.feature_importances_
    indOfSF = indOfSF.argsort()[-NumOfSL:][::-1]
    listOfSF = list()
    for ind in indOfSF:
        listOfSF.append(col[ind])
    return listOfSF
