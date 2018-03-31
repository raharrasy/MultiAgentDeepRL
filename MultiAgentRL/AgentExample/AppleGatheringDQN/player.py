import pygame
import random
import numpy as np
from random import randint
from random import random
from DQN import ConvNet
from ExperienceReplay import ExperienceReplay
from RankBasedExpReplay import RankBasedExpReplay
from WeightBasedExpReplay import WeightBasedExpReplay


class Player(object) :
	def __init__(self,x,y,color = (0,255,0), dead_period = 10, health = 2, batchSize=50, copyIteration = 5000, epsilon = 1.00, discountRate = 0.99, mode = "D-DQN", expDepth = 4, expWidth = 32, expHeight = 42, bufferMaxSize = 100000, alpha = 0.6, beta = 0.8, learningRate = 0.00001, rebalanceFrequency = 60000, weightAdder = 0.00001):
		self.x = x
		self.y = y
		self.color = color
		self.maxHealth = health
		self.mode = mode
		self.action_num = 0
		self.orientation = 0
		self.beam = []
		self.dead_period = dead_period
		self.remaining_time = 0
		self.health = 2
		self.point = 0
		self.maxSize = bufferMaxSize
		self.batchSize = batchSize
		self.is_dead = False
		self.expWidth = expWidth
		self.expHeight = expHeight
		self.prevState = np.zeros((1,self.expWidth,self.expHeight,expDepth))
		self.curState = np.zeros((1,self.expWidth,self.expHeight,expDepth))
		self.playerLastPoint = 0
		self.NN = ConvNet(expWidth, expHeight, expDepth)
		self.copyIteration = copyIteration
		self.action_counter = 0
		self.epsilon = epsilon
		self.discountRate = discountRate
		self.numsOfShots = 0
		self.expDepth = expDepth
		self.alpha = 0
		self.beta = 0
		self.rebalanceFrequency = None
		self.weightAdder = None
		self.learningRate = learningRate
		if "RankExpReplay" in self.mode:
			self.alpha = alpha
			self.beta = beta
			self.rebalanceFrequency = rebalanceFrequency
			self.ExperienceBuffer = RankBasedExpReplay(self.maxSize, self.alpha)
		elif "WeightExpReplay" in self.mode:
			self.alpha = alpha
			self.beta = beta
			if "RUQL-Weight" in self.mode:
				weightAdder = 0
				self.alpha = 1
			self.weightAdder = weightAdder            
			self.ExperienceBuffer = WeightBasedExpReplay(self.maxSize, self.alpha, self.epsilon)
		else:
			self.ExperienceBuffer = ExperienceReplay(self.maxSize)


	def reset(self,location):
		self.x = location[0]
		self.y = location[1]
		self.action_num = 0
		self.point = 0
		self.remaining_time = 0
		self.health = 2
		self.is_dead = False
		self.orientation = 0
		self.numsOfShots = 0
		self.playerLastPoint = 0
		self.prevState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))
		self.curState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))

	def add_player_point(self, point):
		self.playerLastPoint = point
		self.point += point
	
	def setIndex(self,newPosition):
		self.x = newPosition[0]
		self.y = newPosition[1]
	
	def getIndex(self):
		return (self.x,self.y)

	def setDead(self):
		self.health -= 1
		if self.health == 0:
			self.is_dead = True
			self.remaining_time = self.dead_period
	
	def getRemainingTime(self):
		return self.remaining_time
	
	def setAlive(self):
		if self.remaining_time == 0 and self.is_dead:
			self.is_dead = False
			self.health = self.maxHealth
	
	def isDead(self):
		return self.is_dead

	def sense(self,RGBMatrix,ExperienceFlag, LastExpFlag):
		r, g, b = RGBMatrix[:,:,0], RGBMatrix[:,:,1], RGBMatrix[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

		self.prevState = np.copy(self.curState)
		for i in range(0,self.expDepth-1):
                        self.curState[0,:,:,i] = self.curState[0,:,:,i+1]
		self.curState[0,:,:,self.expDepth-1] = gray

		if ExperienceFlag:
			self.ExperienceBuffer.addExperience((np.copy(self.prevState),self.action_num,self.playerLastPoint,np.copy(self.curState),LastExpFlag))

	def learn(self):
		if (self.action_counter % self.batchSize == 0) and (self.action_counter>=self.batchSize):
			sampled_data, samplingWeights, indexList = self.ExperienceBuffer.sample(self.batchSize)
			dataset = np.asarray([a[0][0] for a in sampled_data])
			predictionX = []
			predictionY = []
			takenActions = [a[1] for a in sampled_data]
			rewardList = [a[2] for a in sampled_data]
			nextStates = np.asarray([a[3][0] for a in sampled_data])
			endFlags = [a[4] for a in sampled_data]
			dataY, predictionDifference, probPicked = self.targetCalculation(dataset, rewardList, nextStates, endFlags, takenActions)
			learningWeights = self.weightCalculation(sampled_data)
			if ("RankExpReplay" in self.mode) or ("WeightExpReplay" in self.mode):            
				self.modifyPriorities(predictionDifference, probPicked, indexList)
			self.NN.learn(dataset,dataY, learningWeights)

		if (self.action_counter % self.copyIteration == 0) and (self.action_counter!=0):
			self.NN.copyNetwork()
        
		if "RankExpReplay" in self.mode:
			if (self.action_counter%self.rebalanceFrequency == 0) and (self.action_counter!=0):
				self.ExperienceBuffer.rebalance()
            
	def modifyPriorities(self, predictionDifference, probPicked, indexes):
		if "RUQL-Weight" in self.mode:
			for a in len(indexes):
				self.ExperienceBuffer.modifyExperience(probPicked[a], indexes[a])                
		else:
			for a in len(indexes):
				self.ExperienceBuffer.modifyExperience(predictionDifference[a], indexes[a])

	def act(self):
		self.action_counter += 1
		taken_action = randint(0,7)
		randomVal = random()
		if randomVal > self.epsilon:
			a = self.NN.computeRes(self.curState)
			maxVal = max(a[0])
			maxIndexes = [i for i, j in enumerate(a[0]) if j == maxVal]
			indexChosen = randint(0,len(maxIndexes)-1)
			taken_action = maxIndexes[indexChosen]

		if self.is_dead:
			taken_action = 7
		self.action_num = taken_action

		if taken_action == 4:
			self.numsOfShots += 1

		return self.action_num

	def checkpointing(self, filename, step = 0):
		self.NN.checkpointing(filename,step)
        
	def targetCalculation(self, dataset, rewardList, nextStates, endFlags, takenActions):
			dataset_pred = self.NN.computeRes(dataset)
			res = []
			addition = []      
			if "D-DQN" in self.mode:
				detMax = self.NN.computeRes(nextStates)
				reservoir = []
				for ii in range(self.batchSize):
					eval_res = detMax[ii]
					max_eval = max(eval_res)
					idxMax = [idx for idx,val in enumerate(eval_res) if val == max_eval]
					indexChosen = randint(0,len(idxMax)-1)
					taken_action = idxMax[indexChosen]
					reservoir.append(taken_action)
				targetVal = self.NN.targetCompute(predData)
				for ii in range(self.batchSize):
					addition.append(self.discountRate*targetVal[ii][reservoir[ii]])
			else:
				resPred = self.NN.targetCompute(nextStates)
				addition = [self.discountRate*max(a) for a in resPred]
                
			for a in range(len(addition)):
				if endFlags[a] :
					addition[a] = 0
			res = [c+d for (c,d) in zip(rewardList,addition)]

			predictionY = []
			predictionDifference = []
			probPicked = []            
			for ii in range(self.batchSize):
				initPred = dataset_pred[ii]
				diff = abs(res[ii] - initPred[takenActions[ii]])
				maxPred = max(initPred)
				maxIndexes = [a for (a,b) in enumerate(initPred) if b==maxPred]
				prob = self.epsilon/len(initPred)
				if takenActions[ii] in maxIndexes:
					prob += (1.0-self.epsilon)/len(maxIndexes)
				initPred[takenActions[ii]] = res[ii]
				predictionY.append(initPred)
				predictionDifference.append(diff)
				probPicked.append(prob)

			dataY = np.asarray(predictionY)
			return dataY, predictionDifference, probPicked
        
	def weightCalculation(self, samplingWeights):
		multiplier = None        
		if ("RankExpReplay" in self.mode) or ("WeightExpReplay" in self.mode):
			adjustor = self.size+0.0
			w = np.power(samplingWeights * adjustor, -self.beta)
			w_max = max(w)
			w = np.divide(w, w_max)
			multiplier = np.asarray([[math.sqrt(b*self.learningRate) for b in weights]])
		else:
			multiplier = np.asarray([[math.sqrt(self.learningRate) for b in weights]])
		return multiplier

        
	def printPlayerParams(self):
		print("Player Maximum Health : "+str(self.maxHealth))
		print("Player Width And Height : "+str(self.expWidth)+", "+str(self.expHeight))        
		print("Freeze Intervals : "+str(self.dead_period))
		print("Training Batch Size : "+str(self.batchSize))
		print("Target Network Copy Intervals : "+str(self.copyIteration))
		print("Player Maximum Health : "+str(self.maxHealth))
		print("Discount Rate : "+str(self.discountRate))
		print("Used Algorithm : "+str(self.mode))
		print("Experience Depth : "+str(self.expDepth))
		print("Experience Replay Max Capacity : "+str(self.maxSize))
		print("Learning Rate : "+str(self.learningRate))        
		if ("RankExpReplay" in self.mode) or ("WeightExpReplay" in self.mode):
			print("Initial alpha : "+str(self.alpha))
			print("Initial beta : "+str(self.beta))
		if ("WeightExpReplay" in self.mode):
			print("Weight Adder : "+ str(self.weightAdder))
		if ("RankExpReplay" in self.mode):
			print("Rebalance intervals : "+ str(self.rebalanceFrequency)) 
		print("--------------------------------------------")

