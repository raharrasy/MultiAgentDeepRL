import pygame
import player
import gameplay
import Draw
import food
import numpy as np
from random import sample
import os

class Simulation(object):
	def __init__(self,size,playerNum,maxFoodNum,sight_radius, sight_sideways, mode):
		pygame.init()
		self.size = size
		self.simulationMode = mode
		self.playerNum = playerNum
		self.maxFoodNum = maxFoodNum
		self.sight_radius = sight_radius
		self.sight_sideways = sight_sideways
		self.screen = pygame.Surface((2*(size[0]+2*sight_radius),2*(size[1]+2*sight_radius)),0,32)
		self.state = gameplay.State(self.sight_sideways, self.sight_radius,self.playerNum,self.maxFoodNum,self.size,obsMode=mode)
		self.display = Draw.Display(self.screen,self.state,self.sight_radius)
		self.epsCounter = 0
		self.clock = pygame.time.Clock()
		self.playerTimer = 0
		self.tickWait = 0

	def run(self):
		self.display.drawState()
		self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
		pointsPerEpisode = []
		aggressiveness = []
		self.checkpointing(self.epsCounter)
		while True :
			self.screen.fill((0,0,0))
			self.state.updateState()
			self.display.drawState()
			if self.state.remaining_game_frames == 0:
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)), ExperienceFlag=True, LastExperienceFlag=True)
			self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)), ExperienceFlag=True)
			self.state.learn()
			self.playerTimer = 0
			if self.state.remaining_game_frames == 0:
				playerPoints = [player.point for player in self.state.player_list]
				playerShots = [player.numsOfShots for player in self.state.player_list]
				print(playerPoints)
				print(playerShots)
				self.state.reset()
				self.epsCounter += 1
				self.display.drawState()
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
				self.checkpointing(self.epsCounter)
				if self.epsCounter <= 32:
					epsilon = 1 - (self.epsCounter * (0.95/40))
					self.state.setEpsilon(epsilon)
			if self.epsCounter > 200:
				break

	def checkpointing(self,step):
		counter = 0
		suffix = "../tmp/Apple/DQN1/Player"
		prefix = ".ckpt"
		for player in self.state.player_list:
			player.checkpointing(suffix+str(counter)+prefix, step)
			counter += 1

	def setAgentsList(self,listOfPlayers):
		self.state.setListOfPlayers(listOfPlayers)


if __name__ == '__main__':
	app = Simulation((25,25),4,2,15, 21, "FULL")
	app.state.printGameStatistics()
	random_player_positions = sample(range(len(app.state.coordinate_pairs)), app.playerNum)
	coordinates = [app.state.coordinate_pairs[ii] for ii in random_player_positions]
	player_list = [player.Player(i,j,mode = "DQN",expWidth = 50, expHeight = 50) for (i,j) in coordinates]
	app.setAgentsList(player_list)
	for playr in app.state.player_list:
		playr.printPlayerParams()
	savingDirectory = "../tmp/Apple/DQN1/"
	os.makedirs(savingDirectory)
	app.run()
