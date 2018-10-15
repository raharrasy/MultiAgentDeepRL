from random import randint
from player import Player
from ExperienceReplay import ExperienceReplay
from DQN import DQN
import math
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class DQNPlayer(Player):
	def __init__(self,x,y,player_id,color = (0,255,0), maxExpCapacity = 100000, batchSampleSize = 128, targetUpdates = 10, discountRate = 0.99, expDepth=8, expWidth=32, expHeight=42, epsilonStart = 0, epsilonEnd = 0.99, epsilonDecay = 10000, path="../params"):
		super(DQNPlayer, self).__init__(x,y,player_id,color)
		
		#Initialize maximum experience replay capacity
		self.maxExpCapacity = maxExpCapacity
		self.ExpReplay = ExperienceReplay(self.maxExpCapacity)
		self.id = player_id
		#Initialize batch size
		self.batchSampleSize = batchSampleSize

		#Define device used for computations
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		#Initialize discount rate
		self.discountRate = discountRate

		#Initialize number of concatenated experience
		self.expDepth = expDepth

		#Initialize the size of the observation images
		self.expWidth = expWidth
		self.expHeight = expHeight

		#Initialize Observation Buffers
		self.prevState = np.zeros((1,self.expDepth,self.expWidth,self.expHeight))
		self.curState = np.zeros((1,self.expDepth,self.expWidth,self.expHeight))

		#Initialize target network update frequency
		self.targetUpdates = targetUpdates

		#Initialize policy network and target network
		self.policy_net = DQN(self.expDepth).to(self.device)
		self.target_net = DQN(self.expDepth).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())

		#Number of steps counter
		self.steps_done = 0

		#Set target net to eval only
		self.target_net.eval()

		#Set epsilon for annealing
		self.epsilonStart = epsilonStart
		self.epsilonEnd = epsilonEnd
		self.epsilonDecay = epsilonDecay

		#Initialize type of optimizer
		self.optimizer = optim.RMSprop(self.policy_net.parameters())
		self.savePath = path

	def sense(self,RGBMatrix, ExperienceFlag=False, LastExpFlag=False):
		r, g, b = RGBMatrix[:,:,0], RGBMatrix[:,:,1], RGBMatrix[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

		self.prevState = np.copy(self.curState)
		for i in range(0,self.expDepth-1):
			self.curState[0,i,:,:] = self.curState[0,i+1,:,:]
		self.curState[0,self.expDepth-1,:,:] = gray

		if ExperienceFlag:
			if not LastExpFlag :
				self.ExpReplay.pushItem((np.copy(self.prevState),self.action_num,self.playerLastPoint,
					np.copy(self.curState)))
			else:
				self.ExpReplay.pushItem((np.copy(self.prevState),self.action_num,self.playerLastPoint,
					None))

	def act(self):
		sample = random.random()
		eps_threshold = self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1. * self.steps_done // self.epsilonDecay)
		self.steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
				state = torch.from_numpy(self.curState).double()
				self.action_num = int(self.policy_net(state).max(1)[1].view(1, 1)[0,0].item())
				return self.action_num
		else:
			self.action_num = torch.tensor([[random.randrange(6)]], device=self.device, dtype=torch.long)[0,0].item()
			return self.action_num


	def learn(self):
		if self.ExpReplay.getLength() < self.batchSampleSize:
			return
		transitions = self.ExpReplay.sample(self.batchSampleSize)
			
		states = np.asarray([experience[0][0] for experience in transitions])
		actions = np.asarray([[experience[1]] for experience in transitions])
		rewards = np.asarray([experience[2] for experience in transitions])


		state_batch = torch.from_numpy(states).double()
		action_batch = torch.from_numpy(actions)
		reward_batch = torch.from_numpy(rewards).double()

		non_final_mask = torch.tensor(tuple(map(lambda experience: experience[3] is not None,
                                          transitions)), device=self.device, dtype=torch.uint8)
		non_final_next_states = torch.from_numpy(np.asarray([experience[3][0] for experience in transitions 
			if experience[3] is not None])).double()

		state_action_values = self.policy_net(state_batch).gather(1, action_batch)
		next_state_values = torch.zeros(self.batchSampleSize, device=self.device).double()
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.discountRate) + reward_batch

		# Compute MSE loss
		loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			#Make sure that the gradient doesn't go wild
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		if self.steps_done % self.targetUpdates == 0:
			self.target_net.load_state_dict(self.policy_net.state_dict())


	def checkpointing(self, step):
		finPath = self.savePath+"/"+str(self.id)+"/"+str(step)
		print(finPath)
		directory = os.path.dirname(finPath)
		if not os.path.exists(directory):
			os.makedirs(directory)
		torch.save(self.policy_net.state_dict(), finPath)



		
