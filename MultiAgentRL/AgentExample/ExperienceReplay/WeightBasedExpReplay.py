from Buffer import Buffer
from SumTree import SumTree
import numpy as np

class WeightBasedExpReplay(object):
    def __init__(self,maxSize):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.sumTree = SumTree(self.maxSize)
        self.weights = {}
        self.curSize = 0

    def addExperience(self, experience, weight):
        index = self.buffer.getPointer()
        self.buffer.insert(experience)
        prevWeight = 0
        if index in self.weights:
            prevWeight = self.weights[index]
        diffWeight = weight - prevWeight
        self.weights[index] = weight
        self.sumTree.insert(diffWeight, index)
        self.curSize = min(self.curSize+1,self.maxSize)
        
    def modifyExperience(self, weight, index):
        prevWeight = 0
        if index in self.weights:
            prevWeight = self.weights[index]
        diffWeight = weight - prevWeight
        self.weights[index] = weight
        self.sumTree.insert(diffWeight,index)

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
        
