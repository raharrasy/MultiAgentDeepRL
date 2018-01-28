import pygame
import player
import gameplay
import Draw
import food
import numpy as np
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

		# radius = 10
		# width = (2*radius+1)*2
		# height = width
		# filter = []
		# passed = False

		# for x in range(width):
		# 	ident = int(x/10)
		# 	if ident < radius:
		# 		left_range = -10*ident 
		# 		right_range = ident*10 + 10
		# 		filter = filter + [[x-10*radius,y] for y in range(-10*radius,left_range)]
		# 		filter = filter + [[x-10*radius,y] for y in range(right_range,width-(10*radius))]
		# 	elif ident > radius:
		# 		left_range = -10*(2*radius-ident)
		# 		right_range = 10*(2*radius-ident) + 10
		# 		filter = filter + [[x-10*radius,y] for y in range(-10*radius,left_range)]
		# 		filter = filter + [[x-10*radius,y] for y in range(right_range,width-(10*radius))]

		# newFilter = [[x+100,y+100] for [x,y] in filter]

		# self.filter = newFilter
	def run(self):
		counter = 0
		self.display.drawState()
		self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
		while True :
			self.screen.fill((0,0,0))
			self.state.updateState()
			self.display.drawState()
			self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)),True)
			self.state.learn()
			self.playerTimer = 0
				#splicedArray = pxarray[200:410,200:410]
				#splicedArray[self.filter] = [0,0,0]
				#img = Image.fromarray(splicedArray, 'RGB')
				#img.save('my.png')
				#img.show()
				#splicedArray[self.filter] = [0,0,0]
				#print(splicedArray)
				#print(pxarray[200][100])
					#pxarray = np.asarray(pygame.surfarray.array3d(self.screen))
					#rgbMatrix = [[[x//365//365 % 365, x//365 %365, x%365] for x in y] for y in pxarray]			
				#print(self.state.remaining_game_frames)
			if self.state.remaining_game_frames == 0:
				playerPoints = [player.point for player in self.state.player_list]
				playerShots = [player.numsOfShots for player in self.state.player_list]
				print(playerPoints)
				print(playerShots)
				self.state.reset()
				self.epsCounter += 1
				self.display.drawState()
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
				if self.epsCounter % 5 == 0:
					self.state.checkpointing()
				if self.epsCounter <= 100:
					epsilon = 1 - (self.epsCounter * (0.95/100))
					self.state.setEpsilon(epsilon)
			if self.epsCounter > 600:
				break
		self.state.saveWeights()


if __name__ == '__main__':
	app = Simulation((25,25),4,8,15)
	app.run()

#self.player.update(self.screen)
				#self.Food.update(self.screen,self.player)
				#pxarray = np.asarray(pygame.surfarray.array3d(self.screen))
				#rgbMatrix = [[[x//365//365 % 365, x//365 %365, x%365] for x in y] for y in pxarray]
				
				#self.playerTimer = 0
