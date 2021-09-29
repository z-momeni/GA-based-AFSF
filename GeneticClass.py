#In the Name Of God :)
#

## Imports:
import random
import myFeatureSelection as FS
import calendar
import time
import numpy as np
import pandas as pd

x_data, Y , Xt , Yt ,colName = FS.Preprocces.getPreProcData()


class Individual(object):
    
    steps = [2,3,4] #[1,2,3,4] 
    FilterAlleles = ['P','F','M','R','V','G','A','E','RF']
    chromosome = []
    score = None
    
    def __init__(self, chromosome=None):
        global x_data,Y,Xt,Yt,colName
        self.X = x_data[:]
        self.Y = Y[:]
        self.Xt = Xt[:]
        self.Yt = Yt[:]
        self.col = colName[:]
        self.numOfFeature = self.X.shape[1]
        self.chromosome = chromosome or self._makechromosome()
        self.score = (0,0)  # set during evaluation

        
    def _makechromosome(self):
        "makes a chromosome from randomly selected alleles."
        self.chromosome = []
        
        n_step  = random.choice(self.steps)
        self.chromosome.append(n_step)
        
        rmn_feature = self.numOfFeature

        alpha = int((rmn_feature)/1000)
        
        last_inp = 1
        rate = 10
        for i in range(1,n_step+1):
            random.seed(random.random())
            if rate == 10:
                last_inp = random.randint(last_inp , 30)
            else:
                last_inp = random.randint(last_inp+1 , alpha*rate)
            
            self.chromosome.append(last_inp)
           
            self.chromosome.append(random.choice(self.FilterAlleles))
            if(rate != 1000):
                rate*=10
        self.chromosome = [self.chromosome[0]]+list(reversed(self.chromosome[1:]))
        return self.chromosome
    
    
    def Fitness(self):
        
        Params = {'n_estimators': 400, 'max_depth': 4, 'min_samples_split': 2,
              'subsample':0.6, 'verbose':0, 'warm_start':True, 'alpha':0.6,
              'learning_rate': 0.03, 'loss': 'lad'}
        
        numOfIter = self.chromosome[0]
        ret_f = []
        ret_f = self.col[:]
        X_total = self.X[:]

        for i in range(1,numOfIter*2,2):
            if self.chromosome[i] == 'P':
                ret_f, _ = FS.Filters.myPearson(X_total, self.Y, self.chromosome[i+1], ret_f)
            if self.chromosome[i] == 'F':
                ret_f = FS.Filters.FRegression(X_total, self.Y, self.chromosome[i+1],ret_f)    
            if self.chromosome[i] == 'M':
                ret_f = FS.Filters.MutualInfoRegression(X_total, self.Y, self.chromosome[i+1],ret_f)    
            if self.chromosome[i] == 'R':
                ret_f = FS.Filters.relief(X_total, self.Y, self.chromosome[i+1],ret_f)    
            if self.chromosome[i] == 'V':
                ret_f = FS.Filters.varF(X_total, self.chromosome[i+1],ret_f)    
            if self.chromosome[i] == 'G':
                ret_f = FS.Embeddeds.FIGBR(X_total,self.Y,self.chromosome[i+1],ret_f) 
            if self.chromosome[i] == 'A':
                ret_f = FS.Embeddeds.FIAdaBoost(X_total, self.Y,self.chromosome[i+1],ret_f)      
            if self.chromosome[i] == 'E':
                ret_f = FS.Embeddeds.FIExtra(X_total, self.Y,self.chromosome[i+1],ret_f)   
            if self.chromosome[i] == 'RF':
                ret_f = FS.Embeddeds.FIRF(X_total, self.Y,self.chromosome[i+1],ret_f)   
            
            
        X = X_total.iloc[:,ret_f] 
        xt = self.Xt.iloc[:,ret_f]
        self.score = FS.Evaluations.GBR_with_MAD_CV(X,self.Y,xt,self.Yt,Params,cv=3) 
        return self.score , ret_f
    
    def Crossover(self,other):
        if len(self.chromosome) == len(other.chromosome):
            return self._SameLenght(other)
        
        else:
            return self._DiffLenght(other)

    def _SameLenght(self,other):
        def SortReductions(arr):
            temp = list()
            for num in range(2,len(arr),2):
                temp.append(arr[num])
            temp.sort(reverse = True)
            i = 0
            for num in range(2,len(arr),2):
                arr[num] = temp[i]
                i+=1
            return arr
        child1 , child2 = list([self.chromosome[0]]),list([self.chromosome[0]])
        for i in range(1,(self.chromosome[0]*2),2):
            coin = random.choice([0,1])
            if coin == 1:
                child1 , child2 = self._makeSameLenChildren(other,i,child1,child2) 

            else:
                child2 , child1 = self._makeSameLenChildren(other,i,child2,child1)        
            
        child1 , child2 = SortReductions(child1) , SortReductions(child2)               
        return Individual(child1),Individual(child2)
    
    def _DiffLenght(self,other):
        def makeDiffLenChildren(sh,lng):
            arr1 , arr2 = list() , list()
            
            arr1.append(sh[0])
            arr1 = arr1 + sh[1]
            
            arr2.append(lng[0])
            arr2 = arr2 + lng[2] + lng[1]
            
            return arr1 , arr2
        
        def SortReductions(arr):
            temp = list()
            for num in range(2,len(arr),2):
                temp.append(arr[num])
            temp.sort(reverse = True)
            i = 0
            for num in range(2,len(arr),2):
                arr[num] = temp[i]
                i+=1
            return arr
        
        lenS , lenOth = self.chromosome[0] , other.chromosome[0]
        if lenS > lenOth:
            shortP = [lenOth , self.chromosome[-(lenOth*2):]]
            longP = [lenS , other.chromosome[1:] , self.chromosome[1:((lenS - lenOth)*2)+1]]
        else:
            shortP = [lenS , other.chromosome[-(lenS*2):]]
            longP = [lenOth , self.chromosome[1:] , other.chromosome[1:((lenOth - lenS)*2)+1]]
        
        
        child1 , child2 = makeDiffLenChildren(shortP , longP)
        child1 , child2 = SortReductions(child1) , SortReductions(child2)
        
        return Individual(child1),Individual(child2)
    
    

    def _makeSameLenChildren(self,other,ind,arr1,arr2):
        def PickNumRedForChild(n1,n2):
            if n2 >= n1:
                c1 = random.randint(n1 , n2)
                c2 = random.randint(n1 , n2)
            else:
                c1 = random.randint(n2 , n1)
                c2 = random.randint(n2 , n1)                       
            return c1,c2

        arr1.append(self.chromosome[ind])
        arr2.append(other.chromosome[ind])
        numOfRed1 , numOfRed2 = PickNumRedForChild(self.chromosome[ind+1],other.chromosome[ind+1])
        arr1.append(numOfRed1)
        arr2.append(numOfRed2)    
            
        return arr1,arr2
    
    def Mutation(self,mutation_rate):
        if random.random() < mutation_rate:
                lastVal = self.chromosome[2]
                start = int(max(0.8*lastVal , self.chromosome[4]))
                end = min(int(1.2*lastVal) , self.X.shape[1])
              	temp = random.randint(start, end)
                if temp > 0:
                	self.chromosome[2] = temp
                else:
                  	self.chromosome[2] = 1
        
        for gene in range(4,(self.chromosome[0]*2),2):
            if random.random() < mutation_rate:
                lastVal = self.chromosome[gene]
                start = int(max(0.8*lastVal , self.chromosome[gene+2]))
                end = int(min(1.2*lastVal , self.chromosome[gene-2]))
                temp = random.randint(start , end)
                if temp > 0:
                	self.chromosome[gene] = temp
                else:
                  	self.chromosome[gene] = 1
        
        if self.chromosome[0]==2:
            gene = 2
            
        lastVal = self.chromosome[gene+2]
        temp = random.randint(int(0.8*lastVal) , int(min(1.2*lastVal,self.chromosome[gene])))
        if temp > 0:
        	self.chromosome[gene+2] = temp
        else:
          	self.chromosome[gene+2] = 1
        
        return self
      
      