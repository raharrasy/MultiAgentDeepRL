from Buffer import Buffer
from Heap import Heap
import numpy as np

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

    def addExperience(self, experience, weight):
        index = self.buffer.getPointer()
        self.buffer.insert(experience)
        self.heap.insert(index, weight)
        self.curSize = self.heap.size

    def sample(self, samplesAmount):

        if (self.prevAlpha != self.alpha) or (self.prevSize != self.curSize) :
                self.endPoints, self.weights = computeBoundaries(self.alpha, self.curSize)
                self.prevAlpha = self.alpha
                self.prevSize = self.curSize
        totalWeights = sum(self.weights.tolist())
        startPoint = 0
        expList = []
        weightList = []
        for a in self.endPoints :
                end = a + 1
                diff = end - startPoint 
                sampledNum = np.random.randint(diff, size=1)[0]
                retrIndex = startPoint + sampledNum
                startPoint = end
                expList.append(self.buffer.getItem(self.Heap.getIndex(retrIndex)))
                weightList.append(weightList[retrIndex]/totalWeights)
        return np.asarray(expList),np.asarray(weightList)

    def computeBoundaries(alpha, curSize):
        ranks = list(range(samplesAmount))
        weights = [(1.0/rank+1)**self.alpha for rank in ranks]
        sumAllWeights = sum(weights)
        stops = np.linspace(0,sumAllWeights,curSize+1).tolist()
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
