def __init__(self):
    # Pointers from priority to index
    self.p2i = {}
    # Pointers from index to priority
    self.i2p= {}
    # Pointers from priority to weights
    self.p2w = {}

    # Initial content of heap
    self.p2i[0] = -1
    self.i2p[-1] = 0
    self.p2w[0] = sys.float_info.max

    # Initial size of heap
    self.size = 0

def add(self, index, weight):
    if index in self.i2p :
        self.delete(index)
        self.insert(index,weight)
    else:
        self.insert(index,weight)

def insert(self, index, weight):
    # Add heap size
    self.size += 1
    curPrio = self.size

    # Add experience at the end of the heap
    self.p2i[curPrio] = index
    self.i2p[index] = curPrio
    self.p2w[curPrio] = weight

    # Adjust the order by using the swim operator
    self.swim(curPrio)

def delete(self, index):
    if self.size > 0:
        # Move last element in heap to deleted item 
        changedPrio = self.i2p[index]
        self.p2i[changedPrio] = self.p2i[self.size]
        self.i2p[self.p2i[self.size]] = changedPrio
        self.p2w[changedPrio] = self.p2w[self.size]

        # Delete last element
        del self.i2p[self.p2i[self.size]]
        del self.p2i[self.size]
        del self.p2w[self.size]
        
        self.size -= 1
        self.swim(changedPrio)
        self.sink(changedPrio)

def sink(self, priority):
    children1 = 2*priority + 1
    children0 = 2*priority
    swappedIndex = pickChildren(priority)
    while self.p2w[swappedIndex] > self.p2w[priority]:
        indexPrio = self.p2i[priority]
        weightPrio = self.p2w[priority] 
        # Swap priorities with children
        self.p2i[priority] = self.p2i[swappedIndex]
        self.i2p[priority] = self.i2p[swappedIndex]
        self.p2i[swappedIndex] = indexPrio
        self.i2p[indexPrio] = swappedIndex

        # Swap weights with children
        self.p2w[priority] = self.p2w[swappedIndex]
        self.p2w[swappedIndex] = weightPrio

        #Update pointer
        priority = swappedIndex
        swappedIndex = pickChildren(priority)

    
def pickChildren(self, index):
    swappedIndex = -1
    if (self.p2w[children0] is None) and (self.p2w[children0] is None):
        swappedIndex = -1
    elif self.p2w[children1] is None :
        swappedIndex = children0
    else:
        if self.p2w[children0] > self.p2w[children1] :
            swappedIndex = children0
        else:
            swappedIndex = children1

    return swappedIndex

    

def swim(self, priority):
    parentPrio = priority/2
    while self.p2w[parentPrio] < self.p2w[priority] :
        indexPrio = self.p2i[priority]
        weightPrio = self.p2w[priority] 
        # Swap priorities with parent
        self.p2i[priority] = self.p2i[parentPrio]
        self.i2p[priority] = self.i2p[parentPrio]
        self.p2i[parentPrio] = indexPrio
        self.i2p[indexPrio] = parentPrio

        # Swap weights with parent
        self.p2w[priority] = self.p2w[parentPrio]
        self.p2w[parentPrio] = weightPrio

        #Update pointer
        priority = parentPrio
        parentPrio = priority/2
    
