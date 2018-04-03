import math

class SumTree(object):
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.root = initialize(maxSize)
        self.maxDepth = 0
        multiplier = 1
        self.maxDepth = int(math.ceil(math.log2(maxSize)))

    def getAllSum(self):
        return self.root.getSum()

    def search(self,priority):
        searchNode = self.root
        steps = []
        while searchNode.left or searchNode.right:
            if priority < searchNode.left.getSum():
                searchNode = searchNode.left
                steps.append(0)
            else:
                priority -= searchNode.left.getSum()
                searchNode = searchNode.right
                steps.append(1)
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
            if charS == '0':
                searchNode = searchNode.left
            else:
                searchNode = searchNode.right
        searchNode.sum += priorityDiff

class Node(object):
    def __init__(self, sum=0,right = None, left = None):
        self.sum = sum
        self.right = right
        self.left = left

    def getSum(self):
        return self.sum

    
def initialize(maxSize):
    lst = []
    for a in range(2**math.ceil(math.log2(maxSize))):
        lst.append(Node())
    while len(lst) > 1:
        buffer = []
        for a in range(len(lst)//2):
            aggr = Node(lst[2*a].getSum()+lst[2*a+1].getSum(),lst[2*a+1],lst[2*a])
            buffer.append(aggr)

        lst = buffer
    return lst[0]

