from random import randint

class Buffer(object):
	def __init__(self, maxCapacity):
		self.maxCapacity = maxCapacity
		self.currentLength = 0
		self.expReplay = [None] * self.maxCapacity
		self.pointer = 0

	def pushItem(self,item):
		self.expReplay[self.pointer] = item
		self.pointer = (self.pointer+1)%self.maxCapacity
		if self.currentLength < self.maxCapacity:
			self.currentLength += 1

	def getItem(self,index):
		return self.expReplay[index]

	def getPointer(self):
		return self.pointer

	def getLength(self):
		return self.currentLength