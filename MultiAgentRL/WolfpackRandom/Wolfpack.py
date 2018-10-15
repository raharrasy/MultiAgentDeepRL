import pygame
from DQNPlayer import DQNPlayer
from DDQNPlayer import DDQNPlayer
import gameplay
import Draw
import RandomFood
from random import sample
import numpy as np
from Map import Generator

class Simulation(object):
	def __init__(self,size,playerNum,maxFoodNum,sight_radius, sight_sideways, mode):
		pygame.init()
		self.size = size
		self.playerNum = playerNum
		self.maxFoodNum = maxFoodNum
		self.sight_radius = sight_radius
		self.simulationMode = mode
		self.sight_sideways = sight_sideways
		self.mode = mode
		#self.screen = pygame.Surface((2*(size[0]+2*sight_radius),2*(size[1]+2*sight_radius)),0,32)
		self.screen = pygame.display.set_mode((2*(size[0]+2*sight_radius),2*(size[1]+2*sight_radius)),0,32)
		self.epsCounter = 0
		app = Generator(size,7,8)
		app.initialiseMap()
		app.simulate(2)
		self.map = app.booleanMap
		self.state = gameplay.State(self.sight_sideways,self.sight_radius,self.playerNum,self.maxFoodNum,self.size,self.map,obsMode=self.mode)
		self.display = Draw.Display(self.screen,self.state, self.sight_radius)


	def run(self):
		counter = 0
		self.state.checkpointing(self.epsCounter)
		self.display.drawState()
		self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
		pygame.display.update()
		while True :
			counter +=1
			print(counter)
			self.screen.fill((0,0,0))
			self.state.updateState()
			self.display.drawState()
			if self.state.remaining_game_frames != 1:
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)),ExperienceFlag=True)
			else:
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)),ExperienceFlag=True,lastExpFlag=True)
			pygame.display.update()
			self.state.learn()
			self.playerTimer = 0    
			if self.state.remaining_game_frames == 0:
				playerPoints = [player.point for player in self.state.player_list]
				print(playerPoints)
				print(self.state.wolfPerCapture)
				app = Generator(self.size,7,7)
				app.initialiseMap()
				app.simulate(2)
				self.map = app.booleanMap
				self.state.reset(self.playerNum,self.map)
				self.display.setState(self.state)
				self.display.drawState()
				self.state.sense(np.asarray(pygame.surfarray.array3d(self.screen)))
				self.epsCounter += 1

			if self.epsCounter > 200:
				break

		self.state.saveWeights()

def getFoodCoords(app, numOfFoods):
	foodCoords = []
	possibleCoordinates = list(set(app.state.possibleCoordinates) - set([(player.x,player.y) for player in app.state.player_list]))
	while len(app.state.food_list) < app.state.max_food_num and len(foodCoords) < numOfFoods:
		samplePossibleCoordinates = list(set(possibleCoordinates) - set([(food[0],food[1]) for food in foodCoords]))
		food_positions = sample(range(len(samplePossibleCoordinates)), 1)
		chosenCoordinate = samplePossibleCoordinates[food_positions[0]]
		foodCoords.append(chosenCoordinate)

	return foodCoords


if __name__ == '__main__':
	app = Simulation((25,25),4,2,15, 21, "PARTIAL")
	random_player_positions = sample(range(len(app.state.coordinate_pairs)), app.playerNum)
	coordinates = [app.state.coordinate_pairs[ii] for ii in random_player_positions]
	ids = list(range(4))
	player_list = [DDQNPlayer(i,j,k) for ((i,j),k) in zip(coordinates,ids)]
	foodCoordinates = getFoodCoords(app,2)
	food_list = [RandomFood.RandomFood(i,j) for(i,j) in foodCoordinates]
	print(app.map)
	app.state.setListOfPlayers(player_list)
	app.state.setListOfFoods(food_list)

	app.run()





#self.player.update(self.screen)
				#self.Food.update(self.screen,self.player)
				#pxarray = np.asarray(pygame.surfarray.array3d(self.screen))
				#rgbMatrix = [[[x//365//365 % 365, x//365 %365, x%365] for x in y] for y in pxarray]
				
				#self.playerTimer = 0