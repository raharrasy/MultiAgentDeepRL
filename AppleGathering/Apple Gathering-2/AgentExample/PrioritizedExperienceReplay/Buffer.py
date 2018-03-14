def __init__(self,maxSize):
	self.storage = [None] * maxSize
	self.pointer = 0
	self.maxSize = maxSize

def insert(experience):
	self.storage[self.pointer] = experience
	self.pointer = (self.pointer+1)%maxSize

def getItem(index):
	return self.storage[index]

def getPointer():
	return self.pointer
