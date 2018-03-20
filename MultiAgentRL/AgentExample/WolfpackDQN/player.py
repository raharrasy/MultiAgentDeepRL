import pygame
import random
import numpy as np
from random import randint
from random import random
from DQN import ConvNet
from ExperienceBuffer import ExperienceBuffer

class Player(object) :
	def __init__(self,x,y,color = (0,255,0), updateIteration = 50, batchSize=200, copyIteration = 5000, epsilon = 1.00, discountRate = 0.99, mode = "D-DQN", filename="", expDepth = 4, expWidth = 32, expHeight = 42):
		self.x = x
		self.y = y
		self.color = color		
		self.action_num = 0
		self.point = 0
		self.orientation = 0
		self.prevState = None
		self.curState = None
		self.ExperienceBuffer = ExperienceBuffer()
		self.playerLastPoint = 0
		self.batchSize = batchSize
		self.expWidth = expWidth
		self.expHeight = expHeight
		self.prevState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))
		self.curState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))
		self.NN = ConvNet()
		self.copyIteration = copyIteration
		self.action_counter = 0
		self.epsilon = epsilon
		self.discountRate = discountRate
		self.mode = mode
		self.updateIteration = updateIteration
		self.filename = filename
		self.expDepth = expDepth

	def reset(self,location):
		self.x = location[0]
		self.y = location[1]		
		self.action_num = 0
		self.point = 0
		self.orientation = 0
		self.prevState = None
		self.curState = None
		self.playerLastPoint = 0
		self.prevState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))
		self.curState = np.zeros((1,self.expWidth,self.expHeight,self.expDepth))

	def add_player_point(self, point):
		self.point += point
		self.playerLastPoint = point
	
	def setIndex(self,newPosition):
		self.x = newPosition[0]
		self.y = newPosition[1]
	
	def getIndex(self):
		return (self.x,self.y)
	
	def act(self):
		self.action_counter += 1
		taken_action = randint(0,6)
		randomVal = random()
		if randomVal > self.epsilon:
			a = self.NN.computeRes(self.curState)
			maxVal = max(a[0])
			maxIndexes = [i for i, j in enumerate(a[0]) if j == maxVal]
			indexChosen = randint(0,len(maxIndexes)-1)
			taken_action = maxIndexes[indexChosen]

		self.action_num = taken_action

		return self.action_num

	def sense(self,RGBMatrix, sight_radius, ExperienceFlag):
		r, g, b = RGBMatrix[:,:,0], RGBMatrix[:,:,1], RGBMatrix[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

		self.prevState = np.copy(self.curState)
		for i in range(0,self.expDepth-1):
                        self.curState[0,:,:,i] = self.curState[0,:,:,i+1]
		self.curState[0,:,:,self.expDepth] = gray

		if ExperienceFlag:
			self.ExperienceBuffer.insert((np.copy(self.prevState),self.action_num,self.playerLastPoint,np.copy(self.curState)))


	def learn(self):
		if (self.action_counter >= self.batchSize) and (self.action_counter % self.updateIteration == 0):
			#self.NN.copyNetwork()
			#self.action_counter = 0
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

	def checkpointing(self, filename, step):
		self.NN.checkpointing(filename, step)
