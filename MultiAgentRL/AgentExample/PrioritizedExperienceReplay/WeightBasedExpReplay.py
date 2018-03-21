from Buffer import Buffer
from SumTree import SumTree
import numpy as np

class WeightBasedExpReplay(object):
    def __init__(self,maxSize):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.sumTree = SumTree(self.maxSize)
        self.weights = {}

    def addExperience(self, experience, weight):
        index = self.buffer.getPointer()
        self.buffer.insert(experience)
        prevWeight = 0
        if index in self.weights:
            prevWeight = self.weights[index]
        diffWeight = weight - prevWeight
        self.weights[index] = weight
        self.sumTree.insert(diffWeight, index)

    def sample(self, samplesAmount):
            startPoints = np.linspace(0,self.sumTree.getAllSum(),samplesAmount+1).tolist()
            expList = []
            for a in range(0,len(startPoints)-1) :
                    start = startPoints[a]
                    end = startPoints[a+1]
                    sampledNum = np.random.uniform(start,end)
                    expList.append(self.buffer.getItem(self.sumTree.search(sampledNum)))

            return np.asarray(expList)
