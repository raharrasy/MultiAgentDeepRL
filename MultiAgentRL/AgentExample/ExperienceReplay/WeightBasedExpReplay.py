from Buffer import Buffer
from SumTree import SumTree
from Heap import Heap
import numpy as np

class WeightBasedExpReplay(object):
    def __init__(self,maxSize, alpha = 0.6, epsilon=0.000001):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.sumTree = SumTree(self.maxSize)
        self.weights = {}
        self.alpha = 0.6
        self.curSize = 0
        self.epsilon = epsilon
        self.heap = Heap()

    def addExperience(self, experience):
        weight = self.heap.getMaxPriority()
        index = self.buffer.getPointer()
        self.buffer.insert(experience)
        prevWeight = 0
        diffWeight = weight - prevWeight
        self.weights[index] = weight
        self.sumTree.insert(diffWeight, index)
        self.heap.add(index, weight)
        self.curSize = min(self.curSize+1,self.maxSize)
        
    def modifyExperience(self, weight, index):
        weight = weight + self.epsilon
        weight = weight**self.alpha
        prevWeight = 0
        if index in self.weights:
            prevWeight = self.weights[index]
        diffWeight = weight - prevWeight
        self.weights[index] = weight
        self.sumTree.insert(diffWeight,index)
        self.heap.add(index, weight)

    def sample(self, samplesAmount):
            startPoints = np.linspace(0,self.sumTree.getAllSum(),samplesAmount+1).tolist()
            expList = []
            weightList = []
            indexList = []
            for a in range(0,len(startPoints)-1) :
                    start = startPoints[a]
                    end = startPoints[a+1]
                    sampledNum = np.random.uniform(start,end)
                    retrIndex = self.sumTree.search(sampledNum)
                    expList.append(self.buffer.getItem(retrIndex))
                    weightList.append(self.weights[retrIndex]/self.sumTree.getAllSum())
                    indexList.append(retrIndex)

            return np.asarray(expList),np.asarray(weightList),np.asarray(indexList)
        
    def getMaxPriority(self):
        if self.heap.size == 0:
            return sys.float_info.max
        return self.heap.p2w[1]
        
