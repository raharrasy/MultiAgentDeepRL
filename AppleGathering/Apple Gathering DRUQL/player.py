import pygame
import random
import numpy as np
import tensorflow as tf
from random import randint
from random import random
from DQN import ConvNet
from ExperienceBuffer import ExperienceBuffer
import math

class Player(object) :
	def __init__(self,x,y,color = (0,255,0), dead_period = 10, health = 2, batchSize=50, copyIteration = 5000, epsilon = 1.0, discountRate = 0.99, learningRate = 0.00001):
		self.x = x
		self.y = y
		self.color = color		
		self.maxHealth = health
		self.action_num = 0
		self.orientation = 0
		self.beam = []
		self.dead_period = dead_period
		self.remaining_time = 0
		self.health = 2
		self.ExperienceBuffer = ExperienceBuffer()
		self.point = 0
		self.batchSize = batchSize
		self.is_dead = False
		self.prevState = np.zeros((1,32,42,4))
		self.curState = np.zeros((1,32,42,4))
		self.playerLastPoint = 0
		self.NN = ConvNet()
		self.copyIteration = copyIteration
		self.action_counter = 0
		self.epsilon = epsilon
		self.discountRate = discountRate
		self.chosenProb = 0
		self.learningRate = learningRate
		self.numsOfShots = 0

	def reset(self,location):
		self.x = location[0]
		self.y = location[1]		
		self.action_num = 0
		self.point = 0
		self.remaining_time = 0
		self.health = 2
		self.is_dead = False
		self.orientation = 0
		self.prevState = None
		self.curState = None
		self.playerLastPoint = 0
		self.prevState = np.zeros((1,32,42,4))
		self.curState = np.zeros((1,32,42,4))
		self.numsOfShots = 0
		self.chosenProb = 0

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

	def sense(self,RGBMatrix, sight_radius, ExperienceFlag):
		RGBRep = None
		if self.orientation == 3:
			# Facing left
			x_left = (self.x+(sight_radius)*2) - 20
			x_right = (self.x+ (sight_radius)*2) + 22
			y_up = (self.y+(sight_radius)*2) - 30
			y_down = (self.y+(sight_radius)*2) +2
			RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
			RGBRep = RGBRep.transpose((1,0,2))
			RGBRep = np.fliplr(RGBRep)
		elif self.orientation == 1:
			# Facing right
			x_left = (self.x+(sight_radius)*2) - 20
			x_right = (self.x+ (sight_radius)*2) + 22
			y_up = (self.y+(sight_radius)*2) + 32
			y_down = (self.y+(sight_radius)*2)
			RGBRep = RGBMatrix[x_left:x_right,y_down:y_up]
			RGBRep = RGBRep.transpose((1,0,2))
			RGBRep = RGBRep[::-1]
		elif self.orientation == 0:
			# Facing up
			x_left = (self.x+(sight_radius)*2) - 30
			x_right = (self.x+ (sight_radius)*2) + 2
			y_up = (self.y+(sight_radius)*2) - 20
			y_down = (self.y+(sight_radius)*2) +22
			RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
		elif self.orientation == 2:
			# Facing down
			x_left = (self.x+(sight_radius)*2)
			x_right = (self.x+ (sight_radius)*2) + 32
			y_up = (self.y+(sight_radius)*2) - 20
			y_down = (self.y+(sight_radius)*2) +22
			RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
			RGBRep = np.fliplr(RGBRep)
			RGBRep = RGBRep[::-1]

		r, g, b = RGBRep[:,:,0], RGBRep[:,:,1], RGBRep[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

		self.prevState = np.copy(self.curState)
		self.curState[0,:,:,0] = self.curState[0,:,:,1]
		self.curState[0,:,:,1] = self.curState[0,:,:,2]
		self.curState[0,:,:,2] = self.curState[0,:,:,3]
		self.curState[0,:,:,3] = gray

		if ExperienceFlag and self.action_counter != 0:
			self.ExperienceBuffer.insert((np.copy(self.prevState),self.action_num,self.playerLastPoint,np.copy(self.curState),self.chosenProb))

	def learn(self):
		if (self.action_counter % self.batchSize == 0) and (self.action_counter!=0):
			sampled_data = self.ExperienceBuffer.sample(self.batchSize)
			dataset = np.asarray([a[0][0] for a in sampled_data])
			action_taken = np.asarray([a[1] for a in sampled_data])
			reward = np.asarray([a[2] for a in sampled_data])
			estimator = np.asarray([a[3][0] for a in sampled_data])
			
			multiplier = np.asarray([[math.sqrt(1.0-((1.0-self.learningRate)**(1.0/a[4]))) for a in sampled_data]])
			learningRate = np.transpose(multiplier)

			datasetPred = self.NN.computeRes(dataset)
			chosenVal = self.NN.computeRes(estimator)
			estimatorVal = self.NN.targetCompute(estimator)
			targetValues = []
			for ii in range(len(chosenVal)):
				maximizer = chosenVal[ii]
				max_eval = max(maximizer)
				idxMax = [idx for idx,val in enumerate(maximizer) if val == max_eval]
				indexChosen = randint(0,len(idxMax)-1)
				taken_action = idxMax[indexChosen]
				usedTarget = reward[ii] + self.discountRate*estimatorVal[ii][taken_action] 
				compPred = datasetPred[ii]
				compPred[action_taken[ii]] = usedTarget
				targetValues.append(compPred)

			targetVal = np.asarray(targetValues)

			self.NN.learn(dataset,targetVal,learningRate)

		if (self.action_counter % self.copyIteration == 0) and (self.action_counter!=0):
			self.NN.copyNetwork()		

	def act(self):
		self.action_counter += 1
		taken_action = randint(0,7)
		randomVal = random()
		self.chosenProb = self.epsilon/7
		a = self.NN.computeRes(self.curState)
		maxVal = max(a[0])
		maxIndexes = [i for i, j in enumerate(a[0]) if j == maxVal]
		if randomVal > self.epsilon:
			indexChosen = randint(0,len(maxIndexes)-1)
			taken_action = maxIndexes[indexChosen]
			self.chosenProb += ((1.0-self.epsilon)/len(maxIndexes))

		self.action_num = taken_action
		
		if self.is_dead:
			taken_action = 7
			self.chosenProb = 1.0

		self.action_num = taken_action

		if taken_action == 4:
			self.numsOfShots += 1

		return self.action_num

	def save(self,filename):
		self.NN.save(filename)

	def checkpointing(self, filename):
		self.NN.checkpointing(filename)
