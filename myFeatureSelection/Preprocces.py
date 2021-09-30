import csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler


def getPreProcData():
    
    filename = ['DS097_270','DS236_270','DS242_270','DS317_270',
    		'DS642_270','DS869_270','DS257_270','DS008_270', 
    		'DS149_450','DS169_450','DS638_450','DS812_270',
    		'DS064_450','DS870_450','DS128_450','DS279_450'] 

    XTrain_Frames = []
    YTrain_Frames = []

    for DS in filename:
    #load data:
        pathname = "myFeatureSelection/Datasets/final/"+ DS +".csv"
        X , Y = getDataFromOrginalDS(pathname,xx_range=1, yy_range=1, 
                                                                    xy_range=0, startCol=1, header=0, sep=',') 

        print "DataSet ",DS," : ","[ column:",X.shape[0]," , Row:",X.shape[1],"]"
        X = X.transpose()
        
        XTrain_Frames.append(pd.DataFrame(X)) 
        YTrain_Frames.append(Y)
    
    xtrain_all = pd.concat(XTrain_Frames)
    ytrain_all = pd.concat(YTrain_Frames)
    
    X = Imput(xtrain_all, ytrain_all)
    X = Robust_StandardAll(X)
    X = MinMaxAll(X)
    X = pd.DataFrame(X)
    print "[ column:",X.shape[1]," , Row:",X.shape[0],"]"
    ind = [i for i in range(X.shape[1])]
    #splite-------------------------------------------------
    esm = []
    X["lable"] = ytrain_all.tolist()

    result = X.sort_values(by=['lable'])
    #print result.iloc[:,19335]
    #result.to_csv('out.csv', index = None, header=True)
    Xt = result.iloc[::4,:]

    esm.append(result)
    esm.append(Xt)  

    df = pd.concat(esm)
    x_data = df.drop_duplicates(keep = False)
    #x_data.to_csv('x.csv', index = None, header=True)
    Xt = Xt.sample(frac=1)
    Yt = Xt['lable']
    del Xt['lable']
    x_data = x_data.sample(frac=1)

    Y = x_data['lable']
    del x_data['lable']
    
    #x_data.to_csv('x.csv', index = None, header=True)
    return x_data,Y,Xt,Yt, ind

  
def getPreProcDataSplite():
	######## load train data
    train = ['DS236_270','DS097_270',
             'DS642_270','DS257_270',
             'DS242_270','DS812_270',
             'DS169_450','DS638_450',
             'DS128_450','DS279_450'] 

    XTrain_Frames = []
    YTrain_Frames = []
    print "train:"
    for DS in train:
        pathname = "myFeatureSelection/Datasets/final/"+ DS +".csv"
        X , Y = getDataFromOrginalDS(pathname,xx_range=1, yy_range=1, 
                                     xy_range=0, startCol=1, header=0, sep=',') 

        print "DataSet ",DS," : ","[ column:",X.shape[0]," , Row:",X.shape[1],"]"
        X = X.transpose()
        
        XTrain_Frames.append(pd.DataFrame(X)) 
        YTrain_Frames.append(Y)
    
    xtrain_all = pd.concat(XTrain_Frames)
    ytrain_all = pd.concat(YTrain_Frames)
    
    #X = Imput(xtrain_all, ytrain_all)
    #X = Robust_Standard(xtrain_all)
    #X = MinMax(X)
    
	######## load test data
    test = ['DS317_270','DS869_270','DS008_270',
            'DS870_450','DS149_450','DS064_450']
    XTest_Frames = []
    YTest_Frames = []
    print "test:"
    for DS in test:
        pathname = "myFeatureSelection/Datasets/final/"+ DS +".csv"
        Xt , Yt = getDataFromOrginalDS(pathname,xx_range=1, yy_range=1, 
                                     xy_range=0, startCol=1, header=0, sep=',') 

        print "DataSet ",DS," : ","[ column:",Xt.shape[0]," , Row:",Xt.shape[1],"]"
        Xt = Xt.transpose()
        
        XTest_Frames.append(pd.DataFrame(Xt)) 
        YTest_Frames.append(Yt)
    
    xtest_all = pd.concat(XTest_Frames)
    Ytest_all = pd.concat(YTest_Frames)
    
    ####### standard:
    #X = Imput(xtest_all, Ytest_all)
    X , Xt = Robust_Standard(xtrain_all,xtest_all)
    X , Xt = MinMax(X, Xt)
    
    
    print "train  [ column:",X.shape[1]," , Row:",X.shape[0],"]"
    ind = [i for i in range(X.shape[1])]
    print "test   [ column:",Xt.shape[1]," , Row:",Xt.shape[0],"]"
    
    X , Xt = pd.DataFrame(X) , pd.DataFrame(Xt)
    return X,ytrain_all,Xt,Ytest_all, ind
  

################################################################################################  

def getDataFromOrginalDS(pathName, xx_range, yy_range, xy_range, startCol, header=-1, sep=','):
    mydata = pd.read_csv(pathName,header,sep)
    #for take columns that we want to be in x
    yx_range = []    
    for i in range(startCol,mydata.shape[1]):
        yx_range.append(i)

    #iloc[x_range,y_range]
    """this setting is for data that has class lable in first column, and in first row has index number that 
    # with header = -1 removed."""
    x_data = mydata.iloc[xx_range: , yx_range]
    y_data = mydata.iloc[xy_range , yy_range:]
    return (x_data,y_data)

def MinMaxAll(x_train):
    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    return x_train_minmax

def MinMax(x_train, x_test):
    min_max_scaler = MinMaxScaler().fit(x_train)
    xNormal_train = min_max_scaler.transform(x_train)
    xNormal_test = min_max_scaler.transform(x_test)
    return (xNormal_train,xNormal_test)

def Standard(x_train, x_test):
    scaler = preprocessing.StandardScaler().fit(x_train)
    xNormal_train = scaler.transform(x_train)
    xNormal_test = scaler.transform(x_test)
    return (xNormal_train,xNormal_test)


def Robust_Standard(x_train, x_test):
    scaler = RobustScaler().fit(x_train)
    xNormal_train = scaler.transform(x_train)
    xNormal_test = scaler.transform(x_test)
    return (xNormal_train,xNormal_test)

def Robust_StandardAll(x_train):
    scaler = RobustScaler().fit(x_train)
    xNormal_train = scaler.transform(x_train)
    return (xNormal_train)

def Imput(x,y,strgy = 'mean'):
    imp = SimpleImputer(missing_values=np.nan, strategy=strgy)
    x_impute = imp.fit_transform(x,y)
    return x_impute
    
def overSample(x, y):
    x_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(x, y)
    return (x_resampled, y_resampled)


def split(x, y, sizeSpl):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = sizeSpl)
    return (x_train, x_test, y_train, y_test)


def writeToFile(x, y, pathName, t=0):
    """ for example: pathName = 'dataset/'+filename+'.csv'"""
    myfile = open(pathName,'w')
    writer = csv.writer(myfile)
    """if necessary to transpose: t = 1"""
    if t == 1:
        y = y.transpose()
    firstRow = []
    for c in np.nditer(y):
        if c == [1]:
            firstRow.append(1)
        else:
            firstRow.append(2)
    #print len(firstRow)
    indexOfNewFile = []
    for i in range(x.shape[1]+1):
        indexOfNewFile.append(i)
    #print indexOfNewFile

    writer.writerow(indexOfNewFile)    
    for i in range(x.shape[0]):
        tmp = list()
        tmp.append(firstRow[i])
        
        for j in range(x.shape[1]):
            tmp.append(x[i][j])
        writer.writerow(tmp)