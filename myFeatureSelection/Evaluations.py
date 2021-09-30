import pandas as pd
from math import sqrt
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer


def GBR_with_MAD_CV(x,y,xt,yt,GBR_Params,cv=3):
  def myMAD(y,y_pred):
    temp1 = 0
    y = y.tolist()
    m = len(y)
    for i in range(m):
        dis = y[i] - y_pred[i]
        temp1 = temp1 + abs(dis)
    return temp1/m
  
  estm = ensemble.GradientBoostingRegressor(**GBR_Params)
  
  mad = make_scorer(myMAD)
  
  madCV = cross_val_score(estm , x , y , scoring=mad , cv = cv)
  estm.fit(x,y)
  pred = estm.predict(xt)
  madTest = myMAD(yt,pred)
  return abs(madCV.mean()),madTest





def CrossValGBR(x,y,GBR_Params,CV):
    
    estm = ensemble.GradientBoostingRegressor(**GBR_Params)
    mad = cross_val_score(estm, x, y, scoring =    'neg_mean_absolute_error', cv = CV)
    
    return abs(mad.mean())

def calculator(y,y_pred,m):
    temp1 = 0
    temp2 = 0
    y = y.tolist()
    
    for i in range(m):
        dis = y[i] - y_pred[i]
        temp1 = temp1 + abs(dis)
        temp2 = temp2 + dis**2
    MAD = temp1/m
    MSE = temp2/m
    RMSE = sqrt(MSE)
    Rsquared = r2_score(y, y_pred)
    ans = [MAD,MSE,RMSE,Rsquared]
    return ans
    
def GBR_with_MAD(x,y,xt,yt,GBR_Params):

    clf = ensemble.GradientBoostingRegressor(**GBR_Params)
    clf.fit(x , y)
    #madTrain = mean_absolute_error(y, clf.predict(x))
    pred = clf.predict(x)
    ansTrain = calculator(y,pred,len(pred))

    #mad = mean_absolute_error(yt, clf.predict(xt)) 
    pred = clf.predict(xt)
    ansTest = calculator(yt,pred,len(pred))
    #return abs(mad.mean()),madTrain

    return ansTrain,ansTest
    

def AdaBoosting_with_MAD(x,y,xt,yt,indexF,e_iloc = 1):
    
    if e_iloc:
        x_train = pd.DataFrame(x)
        x_test = pd.DataFrame(xt)

        x_train_filterd = x_train.iloc[0:,indexF]
        x_test_filterd = x_test.iloc[0:,indexF]
    
    else:
        x_train_filterd = x.iloc[0:,indexF]
        x_test_filterd = xt.iloc[0:,indexF]
        
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train_filterd, y)
    
    mad = mean_absolute_error(yt, clf.predict(x_test_filterd))
    
    return mad

def AdaBoosting_with_MAD_CV(x,y,indexF,cv):
    
    x_train = pd.DataFrame(x)

    x_train_filterd = x_train.iloc[0:,indexF]
    
    clf = AdaBoostClassifier(n_estimators=100)
    mad = cross_val_score(clf, x_train_filterd, y, scoring = 'neg_mean_absolute_error', cv = cv)
    
    return abs(mad.mean())