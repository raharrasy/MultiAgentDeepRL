import pygame
import random
from random import randint
from random import random
import numpy as np
from DQN import ConvNet
from ExperienceBuffer import ExperienceBuffer

class Food(object):
	def __init__(self,x,y,color = (255,0,0), dead_period = 15, updateIteration = 50, batchSize=200, copyIteration = 5000, epsilon = 1.00, discountRate = 0.99, mode = "D-DQN"):
		# Ensure that all inputs have been multiplied by 10 beforehand.
		self.x = x
		self.y = y
		self.color = color
		self.action_num = 0
		self.orientation = 0
		self.is_dead = False
		self.dead_period = dead_period
		self.remaining_time = 0
		self.prevState = None
		self.curState = None
		self.playerLastPoint = 0
		self.point = 0
		self.batchSize = batchSize
		self.prevState = np.zeros((1,32,42,4))
		self.curState = np.zeros((1,32,42,4))
		self.NN = ConvNet()
		self.ExperienceBuffer = ExperienceBuffer()
		self.copyIteration = copyIteration
		self.action_counter = 0
		self.epsilon = epsilon
		self.discountRate = discountRate
		self.mode = mode
		self.preventLearning = False
		self.updateIteration = updateIteration
		#self.filename = filename

	def reset(self,location):
		self.x = location[0]
		self.y = location[1]		
		self.action_num = 0
		self.point = 0
		self.remaining_time = 0
		self.is_dead = False
		self.orientation = 0
		self.prevState = None
		self.curState = None
		self.playerLastPoint = 0
		self.prevState = np.zeros((1,32,42,4))
		self.curState = np.zeros((1,32,42,4))

	def add_point(self, point):
		self.point += point
		self.playerLastPoint = point
	
	def setIndex(self,newPosition):
		self.x = newPosition[0]
		self.y = newPosition[1]
	
	def getIndex(self):
		return (self.x,self.y)

	def setAlive(self):
		self.is_dead = False
		self.preventLearning = False
	
	def setDead(self):
		self.is_dead = True
		self.remaining_time = self.dead_period
	
	def getRemainingTime(self):
		return self.remaining_time
	
	def isDead(self):
		return self.is_dead

	def act(self):
		if not self.is_dead:
			self.action_counter += 1
		taken_action = randint(0,6)
		randomVal = random()
		if (not self.is_dead) and randomVal > self.epsilon:
			a = self.NN.computeRes(self.curState)
			maxVal = max(a[0])
			maxIndexes = [i for i, j in enumerate(a[0]) if j == maxVal]
			indexChosen = randint(0,len(maxIndexes)-1)
			taken_action = maxIndexes[indexChosen]

		if self.is_dead:
			taken_action = 4
		self.action_num = taken_action

		return self.action_num

	def sense(self,RGBMatrix,sight_radius,ExperienceFlag=False):		
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

		if ExperienceFlag and (not self.preventLearning):
			self.ExperienceBuffer.insert((np.copy(self.prevState),self.action_num,self.playerLastPoint,np.copy(self.curState)))
			if self.is_dead:
				self.preventLearning = True

	def learn(self):
		if (self.action_counter >= self.batchSize) and (self.action_counter % self.updateIteration == 0):
			sampled_data = self.ExperienceBuffer.sample(self.batchSize)
			dataset = np.asarray([a[0][0] for a in sampled_data])
			dataset_pred = self.NN.computeRes(dataset)
			if self.mode == "DQN":
				predictionX = []
				predictionY = []
				rewardList = [a[2] for a in sampled_data]
				predData = np.asarray([a[3][0] for a in sampled_data])
				resPred = self.NN.targetCompute(predData)
				addition = [self.discountRate*max(a) for a in resPred]
				res = [c+d for (c,d) in zip(rewardList,addition)]
				

				for ii in range(self.batchSize):
					data = sampled_data[ii]
					initPred = dataset_pred[ii]
					initPred[data[1]] = res[ii]
					predictionX.append(data[0][0])
					predictionY.append(initPred)

				dataX = np.asarray(predictionX)
				dataY = np.asarray(predictionY)

				self.NN.learn(dataX,dataY)
			elif self.mode == "D-DQN":
				predictionX = []
				predictionY = []
				rewardList = [a[2] for a in sampled_data]
				predData = np.asarray([a[3][0] for a in sampled_data])
				detMax = self.NN.computeRes(predData)
				reservoir = []
				for ii in range(self.batchSize):
					eval_res = detMax[ii]
					max_eval = max(eval_res)
					idxMax = [idx for idx,val in enumerate(eval_res) if val == max_eval]
					indexChosen = randint(0,len(idxMax)-1)
					taken_action = idxMax[indexChosen]
					reservoir.append(taken_action)

				targetVal = self.NN.targetCompute(predData)
				addition = []
				for ii in range(self.batchSize):
					addition.append(self.discountRate*targetVal[ii][reservoir[ii]])
				res = [c+d for (c,d) in zip(rewardList,addition)]

				for ii in range(self.batchSize):
					data = sampled_data[ii]
					initPred = dataset_pred[ii]
					initPred[data[1]] = res[ii]
					predictionX.append(data[0][0])
					predictionY.append(initPred)

				dataX = np.asarray(predictionX)
				dataY = np.asarray(predictionY)

				self.NN.learn(dataX,dataY)

		if (self.action_counter % self.copyIteration == 0) and (self.action_counter>self.batchSize):
			self.NN.copyNetwork()

	def save(self,filename):
		self.NN.save(filename)


	def checkpointing(self, filename):
		self.NN.checkpointing(filename)

	def checkpointing2(self, filename, step=0):
		self.NN.checkpointing2(filename, step)
