def __init__(self, maxSize):
	self.maxSize = maxSize
	self.root = initialize(maxSize)
	self.maxDepth = 0
	multiplier = 1
	while multiplier <= self.maxSize:
		self.maxDepth+=1
		multiplier *= 2
	initialize(maxSize)

def getAllSum(self):
        return self.root.getSum()

def initialize(maxSize):
	lst = [Node() for i in range(maxSize)]
	while len(lst) > 1:
		buffer = []
		for a in range((len(lst)/2)-1):
			aggr = Node(lst[2*a].getSum()+lst[2*a+1].getSum(),lst[2*a+1],lst[2*a])
			buffer.append(aggr)
		if len(lst) % 2 == 0:
			aggr = Node(lst[2*a].getSum()+lst[2*a+1].getSum(),lst[2*a+1],lst[2*a])
                        buffer.append(aggr)
		else:
			aggr = Node(sum=lst[2*a].getSum()+lst[2*a+1].getSum(),left=lst[2*a])
                        buffer.append(aggr)

		lst = buffer
	

def search(self,priority):
	searchNode = self.root
	steps = []
	while searchNode.left or searchNode.right:
		if priority < searchnode.getSum():
			searchNode = searchNode.left
			steps = steps.append(0)
		else:
			searchNode = searchNode.right
			steps = steps.append(1)
	length = len(steps)
	multiplier = [2**(length-i-1) for i in range(length)]
	return sum([i*j for (i, j) in zip(steps, multiplier)])

def insert(self,priorityDiff, index):
	initial = "{0:b}".format(index)
	adder = "0" * (self.maxDepth-len(initial))
	searchString = adder+initial

	searchNode = self.root
	for a in range(len(searchString)):
		charS = searchString[a]
		searchNode.sum += priorityDiff
		if charS == 0:
			searchNode = searchNode.left
		else:
			searchNode = searchNode.right
	searchnode.sum += priorityDiff

class Node:
	def __init__(self, sum=0,right = None, left = None):
		self.sum = sum
		self.right = right
		self.left = left

	def getSum():
		return self.sum


