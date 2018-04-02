from Buffer import Buffer
from Heap import Heap
import numpy as np
import sys

class RankBasedExpReplay(object): 
    def __init__(self,maxSize, alpha=0.6):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.heap = Heap()
        self.weights = None

        #Add two flags to indicate whether alpha or queue size has changed
        self.prevAlpha = alpha
        self.prevSize =0

        # Variables to store current alpha and exp replay size
        self.alpha = alpha
        self.curSize = 0

        #Weightings to each experience
        self.endPoints = []

    def addExperience(self, experience):
        index = self.buffer.getPointer()
        self.buffer.insert(experience)
        weight = self.heap.getMaxPriority()
        self.heap.add(index, weight)
        self.curSize = self.heap.size
        
    def modifyExperience(self, weight, index):
        self.heap.add(index, weight)
        self.curSize = self.heap.size
        
    def sample(self, samplesAmount):

        if (self.prevAlpha != self.alpha) or (self.prevSize != self.curSize) :
                self.endPoints, self.weights = self.computeBoundaries(self.alpha, self.curSize, samplesAmount)
                self.prevAlpha = self.alpha
                self.prevSize = self.curSize
        totalWeights = sum(self.weights)
        startPoint = 0
        expList = []
        weightList = []
        indexList = []
        for a in self.endPoints :
                end = a + 1
                diff = end - startPoint 
                sampledNum = np.random.randint(diff, size=1)[0]
                retrIndex = startPoint + sampledNum
                startPoint = end
                expList.append(self.buffer.getItem(self.Heap.getIndex(retrIndex)))
                weightList.append(weightList[retrIndex]/totalWeights)
                indexList.append(retrIndex)
        return np.asarray(expList),np.asarray(weightList),np.asarray(indexList)

    def computeBoundaries(self, lpha, curSize, samplesAmount):
        ranks = list(range(curSize))
        weights = [(1.0/(rank+1))**self.alpha for rank in ranks]
        sumAllWeights = sum(weights)
        stops = np.linspace(0,sumAllWeights,samplesAmount+1).tolist()
        del stops[0]
        curSum = 0
        curFounded = 0
        curStop = -1
        results = []
        for a in weights:
                curSum += a
                curStop += 1
                if curSum >= stops[curFounded]:
                        results.append(curStop)
                        curFounded += 1

        return results, weights
    
    def rebalance(self):
        indexList = []
        weightList = []
        while self.heap.size != 0:
            maxIndex = self.heap.p2i[1]
            maxWeight = self.heap.p2w[1]
            indexList.append(maxIndex)
            weightList.append(maxWeight)
            self.heap.delete(maxIndex)
        for a in range(len(indexList)):
            self.add(indexList[a],weightList[a])
            
    def getMaxPriority(self):
        if self.heap.size == 0:
            return sys.float_info.max
        return self.heap.p2w[1]
            
