import Evaluations


def SFS_with_GBR(x,y,listOfIndex,params,cv):

    bestIndex =[]
    lengh = len(listOfIndex)
    bestMAD = 1000
    
    for i in range(lengh):
        tempIndex = -1
        for j in range(lengh):
            temp = []
            temp = bestIndex[:]
            if listOfIndex[j] != -1:
                temp.append(listOfIndex[j])
                xtemp = x.iloc[:,temp]
                tempMAD = Evaluations.GBR_with_MAD_CV(xtemp,y,params,cv)
                #print "tempMAD:",tempMAD,"\n"
                if tempMAD < bestMAD :
                    bestMAD = tempMAD
                    print bestMAD
                    tempIndex = j

        if tempIndex != -1 :
            bestIndex.append(listOfIndex[tempIndex])
            listOfIndex[tempIndex] = -1 
    
    return (bestIndex,bestMAD)

def SFS_with_AdaBoosting(x,y,listOfIndex,cv):

    bestIndex =[]
    lengh = len(listOfIndex)
    bestMAD = 1000
    
    for i in range(lengh):
        tempIndex = -1
        for j in range(lengh):
            temp = []
            temp = bestIndex[:]
            if listOfIndex[j] != -1:
                temp.append(listOfIndex[j])
                tempMAD = Evaluations.AdaBoosting_with_MAD_CV(x,y,temp,cv)
                #print "tempMAD:",tempMAD,"\n"
                if tempMAD < bestMAD :
                    bestMAD = tempMAD
                    print bestMAD
                    tempIndex = j

        if tempIndex != -1 :
            bestIndex.append(listOfIndex[tempIndex])
            listOfIndex[tempIndex] = -1 
    
    return (bestIndex,bestMAD)