import GeneticClass as GN
from numpy.random import choice
import matplotlib.pyplot as plt
import operator
import random

def InitialPopulation(number):
    population = list()
    for i in range(number):
        individual = GN.Individual()
        population.append(individual)
    return population
  
def InitialSpecifyPopulation(number):
    pop = [[3,'P', 12241 , 'A', 4456, 'RF' , 20],
           [3, 'M', 10874, 'R', 1618, 'R', 38],
           [3,'M', 11313 , 'G', 3218, 'RF' , 18],
           [3, 'R', 14132 , 'A', 2032 , 'G' , 25],
           [3,'R', 12241 , 'A', 4456, 'G' , 20],
           [3,'R', 7825, 'A', 3465, 'G' , 24]]
                                                                                    
    population = list()
    for i in range(len(pop)):
        individual = GN.Individual(pop[i])
        population.append(individual)

    for i in range(number-len(pop)):
        individual = GN.Individual()
        population.append(individual)
    return population


def FitnessSorted(individuals):
    fitness = list()
    n = 1
    for individual in individuals:
        print n," )",individual.chromosome
        fit , selectedF = individual.Fitness()
        fitness.append(fit[0])
        print "on train: "
        print fit[0]
        print "on test: "
        print fit[1]
        print "Selected Feature: ",selectedF
        print
        n+=1 
      
    dict_indiFit = dict(zip(individuals[:],fitness[:]))
    return sorted(dict_indiFit.items(), key=operator.itemgetter(1))
  

def FitnessSortedChild(individuals):
    fitness = list()
    n = 1
    for individual in individuals:
        print n," child )",individual.chromosome
        fit , selectedF = individual.Fitness()
        fitness.append(fit[0])
        print "on train: "
        print fit[0]
        print "on test: "
        print fit[1]
        print "Selected Feature: ",selectedF
        print
        n+=1
      
    dict_indiFit = dict(zip(individuals[:],fitness[:]))
    return sorted(dict_indiFit.items(), key=operator.itemgetter(1))
  

def CalProbability(dic_individualFitness):
    probability , prob = list() , list()
    total = 0
    key = list()
    for item in dic_individualFitness:
        key.append(item[0])
        total = total + (1/item[1])
        prob.append(1/item[1])

    for fit in prob:
        probability.append(fit/total)
    dict_indiProb = dict(zip(key,probability))
    # sorted = sorted(dict_indiProb.items(), key=operator.itemgetter(1) , reverse = True)
    return dict_indiProb


def Selection(popRanked, eliteSize):
    selectionResults = []
    probs = CalProbability(popRanked)
    bestParent = []
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
        bestParent.append(popRanked[i])
    remain = choice(probs.keys(),len(popRanked) - eliteSize , p = probs.values())
    
    for i in range(len(popRanked) - eliteSize):
        selectionResults += [remain[i]]
    return selectionResults,bestParent

def getKey(item):
    return item[1]
  
def BreedPopulation(matingpool, bestparent, eliteSize, mutationRate):
    childrens = []
    length = len(matingpool)
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        childrens.append(bestparent[i])
    tempChild = list()
    for i in range(0, length,2):
        child1 , child2 = pool[i].Crossover(pool[i+1])
        tempChild.append(child1)
        tempChild.append(child2)
    mutateChild = MutatePopulation(tempChild, mutationRate)
    childRank = FitnessSortedChild(mutateChild)
    temp = childRank[:length-eliteSize]
    for item in temp:
        childrens.append(item)
      
    nextGeneration = sorted(childrens, key=getKey)
    return nextGeneration


def MutatePopulation(population, mutationRate):
    mutatedPop = []

    for indi in population:
        mutatedInd = indi.Mutation(mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def NextGeneration(currentGen, eliteSize, mutationRate,k):
    popRanked = currentGen
    matingpool , bestParent = Selection(popRanked, eliteSize)
    print k," generation"
    popRankedNext = BreedPopulation(matingpool, bestParent,eliteSize, mutationRate)
    return popRanked,popRankedNext
  

def FirstGeneration(currentGen, eliteSize, mutationRate):
    popRanked = FitnessSorted(currentGen)
    matingpool , bestParent = Selection(popRanked, eliteSize)
    print 2,' childeren generation'
    popRankedNext = BreedPopulation(matingpool, bestParent,eliteSize, mutationRate)
    return popRanked,popRankedNext


def geneticAlgorithmPlot(popSize, eliteSize, mutationRate, generations):
    #pop = InitialPopulation(popSize)
    pop = InitialSpecifyPopulation(popSize)
    progressBest = []
    progressWorst = []
    progressMean = []
    print 1," generation"
    fitness , current = FirstGeneration(pop, eliteSize, mutationRate)
    n = 1
    print "****************************************************************"
    for indi in current:
        print n,"  ) ",indi[1], "  ---> ",indi[0].chromosome
        n+=1
    progressBest.append(fitness[0][1])
    progressWorst.append(fitness[-1][1])
    sum = 0
    for i in range(0, popSize):
        sum += fitness[i][1]
    mean = sum / popSize
    progressMean.append(mean)
    x = [1]
    
    for i in range(2, generations+1):
        x.append(i)
        fitness , current = NextGeneration(current, eliteSize, mutationRate,i)
        n = 1
        print "****************************************************************"
        for indi in current:
            print n,"  ) ",indi[1], "  ---> ",indi[0].chromosome
            n+=1
    
        progressBest.append(fitness[0][1])
        progressWorst.append(fitness[-1][1])
        sum = 0
        for i in range(0, popSize):
            sum += fitness[i][1]
        mean = sum / popSize
        progressMean.append(mean)

        print "#################################################################"
    

    progressBest.append(current[0][1])
    progressWorst.append(current[-1][1])
    sum = 0
    for i in range(0, popSize):
        sum += current[i][1]
    mean = sum / popSize
    progressMean.append(mean)

    print "progressBest = ",progressBest
    print "progressWorst = ",progressWorst
    print "progressMean = ",progressMean
    plt.plot(x , progressBest , color='green') 
    plt.ylabel('MAD')   
    plt.xlabel('Generation')
    plt.plot(x , progressWorst , color='red') 
    plt.ylabel('MAD')
    plt.xlabel('Generation')
    plt.plot(x , progressMean , color='blue') 
    plt.ylabel('MAD')
    plt.xlabel('Generation')
    plt.show();
    plt.savefig('result.png')
    bestObj = current[0][0]
    bestMethod = bestObj.chromosome
    bestMAD = current[0][1]
    return bestMethod,bestMAD



bestInd , bestMAD = geneticAlgorithmPlot(popSize=30, eliteSize=4, mutationRate=0.38, generations=50)
print "best chromosom in final generation: " ,bestInd 
print "Best MAD: ", bestMAD
