from random import sample
from Buffer import Buffer
class ExperienceReplay(object):
	def __init__(self, maxCapacity):
		self.maxCapacity = maxCapacity
		self.buffer = Buffer(self.maxCapacity)

	def pushItem(self,item):
		self.buffer.pushItem(item)

	def sample(self, batchSize):
		availableIndexes = self.getLength()
		sampledIndexes = sample(range(0, availableIndexes), batchSize)
		return [self.buffer.getItem(index) for index in sampledIndexes]

	def getLength(self):
		return self.buffer.getLength()