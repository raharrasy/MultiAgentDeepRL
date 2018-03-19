import random
import copy



class Generator(object):
        """
                A class that generates maps for the Wolfpack game.
                Difficulties of the map can be regulated through the
                deathLimit and birthLimit parameters.
        """
	def __init__(self, size, deathLimit = 4, birthLimit = 3):
		self.x_size = size[0]
		self.y_size = size[1]
		self.booleanMap = [[False]*k for k in [self.x_size] * self.y_size]
		self.probStartAlive = 0.82;
		self.deathLimit = deathLimit
		self.birthLimit = birthLimit
		self.copy = None
 
	def initialiseMap(self):
		for x in range(self.x_size):
			for y in range(self.y_size):
				if random.random() < self.probStartAlive :
					self.booleanMap[y][x] = True;	

	def doSimulationStep(self):
		
		newMap = [[False]*k for k in [self.x_size] * self.y_size]
		for x in range(self.x_size):
			for y in range(self.y_size):
				alive = self.countAliveNeighbours(x, y)
				if self.booleanMap[y][x] :
					if alive < self.deathLimit :
						newMap[y][x] = False
					else:
						newMap[y][x] = True
				else:
					if alive > self.birthLimit :
						newMap[y][x] = True
					else :
						newMap[y][x] = False
		self.booleanMap = newMap

	def countAliveNeighbours(self,x,y):
		count = 0
		for i in range(-1,2):
			for j in range(-1,2):
				neighbour_x = x+i
				neighbour_y = y+j
				if not ((i == 0) and (j == 0)):
					if neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= self.x_size or neighbour_y >= self.y_size :
						count = count + 1
					elif self.booleanMap[neighbour_y][neighbour_x]:
						count = count + 1;
		return count

	def simulate(self, numSteps):
		done = False
		while not done:
			self.booleanMap = [[False]*k for k in [self.x_size] * self.y_size]
			self.initialiseMap()
			for kk in range(numSteps):
				self.doSimulationStep()

			if self.doFloodfill(self.booleanMap) :
				done = True

	def doFloodfill(self,newMap):
		self.copy = copy.deepcopy(newMap)
		foundX, foundY = -1,-1
		for i in range(len(self.copy)):
			flag = False
			for j in range(len(self.copy[i])):
				if not self.copy[i][j]:
					foundX = i
					foundY = j
					flag = True
					break
			if flag:
				break
		self.floodfill(foundX,foundY)
		done = True
		for i in range(len(self.copy)):
			flag = False
			for j in range(len(self.copy[i])):
				#print(self.copy[i][j])
				if not self.copy[i][j]:
					done = False
					flag = True
					break
			if flag:
				break
		return done


	def floodfill(self,x,y):
		queue = []
		queue.append((x,y))
		while len(queue) != 0:
			a = queue[0][0]
			b = queue[0][1]

			del queue[0]
			if not self.copy[a][b]:
				self.copy[a][b] = True
			
			if (not a+1>=len(self.copy)) and (not self.copy[a+1][b]):
				queue.append((a+1,b))
				self.copy[a+1][b] = True
			if (not (a-1<0)) and (not self.copy[a-1][b]):
				queue.append((a-1,b))
				self.copy[a-1][b] = True
			if (not b+1>=len(self.copy[0])) and (not self.copy[a][b+1]):
				queue.append((a,b+1))
				self.copy[a][b+1] = True
			if (not b-1<0) and (not self.copy[a][b-1]):
				queue.append((a,b-1))
				self.copy[a][b-1] = True

			#print(queue)
		

