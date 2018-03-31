from Buffer import Buffer
from Heap import Heap
import numpy as np

class ExperienceReplay(object): 
    def __init__(self,maxSize, alpha=0.6):
        self.maxSize = maxSize
        self.buffer = Buffer(self.maxSize)
        self.curSize = 0

    def addExperience(self, experience):
        self.buffer.insert(experience)
        self.curSize = min(self.curSize+1,self.maxSize)

    def sample(self, samplesAmount):
        sampledPoints = np.random.choice(self.curSize, samplesAmount, replace=False).tolist()
        expList = []
        for a in sampledPoints :
                expList.append(self.buffer.getItem(a))
        return np.asarray(expList), None, None