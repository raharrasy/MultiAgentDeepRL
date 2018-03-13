import pygame
import player
import gameplay
import Draw
import food
import numpy as np


class Simulation(object):
	def __init__(self,size,playerNum,maxFoodNum,sight_radius):
		pygame.init()
		self.size = size
		self.playerNum = playerNum
		self.maxFoodNum = maxFoodNum
		self.sight_radius = sight_radius
		self.screen = pygame.Surface((2*(size[0]+2*sight_radius),2*(size[1]+2*sight_radius)),0,32)
		self.state = gameplay.State(self.sight_radius,self.playerNum,self.maxFoodNum,self.size)
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
			self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)),True)
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

	def checkpointing(self):
			counter = 0
			suffix = "../tmp/Apple/DDQN/Player"
			prefix = ".ckpt"
			for player in self.state.player_list:
				player.checkpointing(suffix+str(counter)+prefix)
				counter += 1
		#		counter += 1


if __name__ == '__main__':
	app = Simulation((25,25),4,8,15)
	app.run()
