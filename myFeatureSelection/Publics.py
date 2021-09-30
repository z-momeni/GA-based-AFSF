
def InterSec(myArr):
    interSec = list(set(myArr[0]) & set(myArr[1]))
    i = 2
    while (i<(len(myArr))):
        interSec = list(set(myArr[i]) & set(interSec))
        #print len(interSec)
        i+=1
    return interSec

def K_frequent(Arr,k):
    temp = []
    for i in range(len(Arr)):
        temp = temp + Arr[i]
    dic_counter = Counter(temp)
    k_most = sorted(dic_counter, key=dic_counter.get, reverse=True)[:k]
    return k_most