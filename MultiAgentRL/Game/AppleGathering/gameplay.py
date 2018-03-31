from random import randint
from random import sample
import numpy as np
import pygame
import player
import food

class State(object) :
		def __init__(self, sight_sideways, sight_radius, num_players, max_food_num, arena_size, food_rate = 15, num_game_frames = 60000,obsMode = "PARTIAL"):
			self.obsMode = obsMode
			self.x_size = arena_size[0]
			self.y_size = arena_size[1]
			self.num_players = num_players            
			self.max_food_num = max_food_num
			self.coordinate_pairs = [(i*2,j*2) for i in range(1,arena_size[0]-1) for j in range(1,arena_size[1]-1)]
			self.player_list = []
			self.food_list = []
			self.beamed_list = []
			self.food_rate = food_rate
			self.waiting_time = self.food_rate
			self.perEpisodeFrame = num_game_frames            
			self.remaining_game_frames = num_game_frames
			self.sight_radius = sight_radius
			self.sight_sideways = sight_sideways
			self.RGBMatrix = None

		def reset(self, food_rate = 15, num_game_frames = 60000):
			random_player_positions = sample(range(len(self.coordinate_pairs)), len(self.player_list))
			coordinates = [self.coordinate_pairs[ii] for ii in random_player_positions]
			ii = 0
			for player in self.player_list:
				player.reset(coordinates[ii])
				ii += 1

			self.food_list = []
			self.beamed_list = []

			self.food_rate = food_rate
			self.waiting_time = self.food_rate
			self.remaining_game_frames = self.perEpisodeFrame
		
		def updateState(self):
			collectiveAction = self.getCollectiveAct()
			self.update_state(collectiveAction)

		def learn(self):
			for player in self.player_list:
				player.learn()
		
		def getCollectiveAct(self):
			collectiveAct = [player.act() for player in self.player_list]
			return collectiveAct
		
		def revive(self):
			for player in self.player_list:
				player.setAlive()

		def sense(self,newRGB,ExperienceFlag=False, LastExperienceFlag = False):
			self.RGBMatrix = newRGB
			RGBObservation = None
			for player in self.player_list:
				if self.obsMode == "PARTIAL":
					RGBObservation = self.computeObservation(self.RGBMatrix, player)
				else:
					RGBObservation = self.computeFullObservation(self.RGBMatrix, player)
				player.sense(RGBObservation,ExperienceFlag)

		def update_state(self,collectiveAct):
			self.waiting_time -= 1
			self.remaining_game_frames -= 1
			self.revive()
			prev_player_position = [(player.x,player.y) for player in self.player_list]
			prev_player_orientation = [player.orientation for player in self.player_list]
			
			#Calculate new player positions
			update_results = self.calculate_new_position(collectiveAct,prev_player_orientation,prev_player_position)
			post_player_position = [(a,b) for (a,b,c) in update_results]
			post_player_orientation = [c for (a,b,c) in update_results]
			for ii in range(len(self.player_list)):
				self.player_list[ii].orientation = post_player_orientation[ii]
			
			#Calculate beam locations
			beamed_locations = self.calculate_beamed_locations(collectiveAct,prev_player_orientation, prev_player_position)
			#Calculate player intersection
			a, seen, result = post_player_position, set(), {}
			for idx, item in enumerate(a):
				if item not in seen:
					result[item] = [idx]
					seen.add(item)
				else:
					#listRes = result[item]
					result[item].append(idx)

			groupings = list(result.values())
			doubles = [t for t in groupings if len(t) >= 2]
			res = set([item for sublist in doubles for item in sublist])
			for ii in range(len(self.player_list)):
				if ii not in res:
					self.player_list[ii].setIndex(post_player_position[ii])
				else:
					self.player_list[ii].setIndex(prev_player_position[ii])
			# Calculate player points and food status
			self.update_food_status(self.player_list)
			#Find players that were shot by beams
			beamed_locations = [item for sublist in beamed_locations for item in sublist]
			beamed_locations = set(beamed_locations)
			self.beamed_list = beamed_locations
			self.update_status(beamed_locations)

			if self.waiting_time == 0:
				self.waiting_time = self.food_rate
				self.addFood()

		def calculate_new_position(self, collectiveAct, prev_player_orientation, prev_player_position):
			zipped_data = list(zip(collectiveAct, prev_player_orientation, prev_player_position))
			result = [self.calculate_indiv_position(a,d,(b,c)) for (a,d,(b,c)) in zipped_data]
			return result
		
		def calculate_indiv_position(self,action,orientation,pair):
			x = pair[0]
			y = pair[1]
			next_x = x
			next_y = y

			# y is kiri kanan
			# x is atas bawah

			# go forward
			if action == 0:
				#Facing upwards
				if orientation == 0:
					next_x -= 2
				#Facing right
				elif orientation == 1:
					next_y += 2
				#Facing downwards
				elif orientation == 2:
					next_x += 2
				else:
					next_y -= 2

				if (next_x< self.x_size*2-2) and (next_x>=2) and (next_y< self.y_size*2-2) and (next_y>= 2):
				 	return (next_x,next_y,orientation)
				else:
				 	return (x,y,orientation)
			# Step right
			elif action == 1:
				#Facing upwards
				if orientation == 0:
					next_y += 2
				#Facing right
				elif orientation == 1:
					next_x += 2
				#Facing downwards
				elif orientation == 2:
					next_y -= 2
				else:
					next_x -= 2

				if (next_x< self.x_size*2-2) and (next_x>=2) and (next_y< self.y_size*2-2) and (next_y>= 2):
					return (next_x,next_y,orientation)
				else:
					return (x,y,orientation)
			#Step back
			elif action == 2:
				#Facing upwards
				if orientation == 0:
					next_x += 2
				#Facing right
				elif orientation == 1:
					next_y -= 2
				#Facing downwards
				elif orientation == 2:
					next_x -= 2
				else:
					next_y += 2

				if (next_x< self.x_size*2-2) and (next_x>=2) and (next_y< self.y_size*2-2) and (next_y>= 2):
				 	return (next_x,next_y,orientation)
				else:
				 	return (x,y,orientation)
			#Step left
			elif action == 3:
				#Facing upwards
				if orientation == 0:
					next_y -= 2
				#Facing right
				elif orientation == 1:
					next_x -= 2
				#Facing downwards
				elif orientation == 2:
					next_y += 2
				else:
					next_x += 2

				if (next_x< self.x_size*2-2) and (next_x>=2) and (next_y< self.y_size*2-2) and (next_y>= 2):
				 	return (next_x,next_y,orientation)
				else:
				 	return (x,y,orientation)
			#beam
			elif action == 4:
				return (x,y,orientation)
			#rotate left
			elif action == 5:
				new_orientation = 0
				if orientation == 0:
					new_orientation = 3
				elif orientation == 1:
					new_orientation = 0
				elif orientation == 2:
					new_orientation = 1
				else:
					new_orientation = 2

				return (x,y,new_orientation)
			#rotate right
			elif action == 6:
				new_orientation = 0
				if orientation == 0:
					new_orientation = 1
				elif orientation == 1:
					new_orientation = 2
				elif orientation == 2:
					new_orientation = 3
				else:
					new_orientation = 0

				return (x,y,new_orientation)
			else:
				return (x,y,orientation)
			

		
		def update_food_status(self,player_list):
			set_of_food_location = set([(food.x,food.y) for food in self.food_list])
			for player in player_list:
				if (player.getIndex() in set_of_food_location) and (not player.isDead()):
					player.add_player_point(1)
					set_of_food_location.remove(player.getIndex())
				else:
					player.add_player_point(0)

			food_list = list(set_of_food_location)
			self.food_list = [food.Food(a[0],a[1]) for a in food_list]

		
		def update_status(self,beamed_locations):
			for player in self.player_list:
				if (not player.isDead()) and (player.getIndex() in beamed_locations):
					player.setDead()
				elif player.isDead():
					player.remaining_time -= 1

		def calculate_beamed_locations(self,action,prev_player_orientation,prev_player_position):
			zipped_data = list(zip(action,prev_player_orientation,prev_player_position))
			shooters = [t for t in zipped_data if t[0] == 4]
			beamed_locations = [self.calculate_indiv_beams(action,orientation,prev_player_position) for (action,orientation,prev_player_position) in shooters]
			return beamed_locations
		
		def calculate_indiv_beams(self,action,prev_player_orientation,prev_player_position):
			beam = []
			if prev_player_orientation == 3:
				beam = [(prev_player_position[0],y*2) for y in range(1,int(prev_player_position[1]/2))]
			elif prev_player_orientation == 1:
				beam = [(prev_player_position[0],y*2) for y in range(int((prev_player_position[1]/2))+1,self.y_size-1)]
			elif prev_player_orientation == 0:
				beam = [(x*2,prev_player_position[1]) for x in range(1,int(prev_player_position[0]/2))]
			elif prev_player_orientation == 2:
				beam = [(x*2,prev_player_position[1]) for x in range(int((prev_player_position[0]/2))+1,self.x_size-1)]
			return beam
		
		def addFood(self):
			if len(self.food_list) < self.max_food_num:
				possibleCoordinates = list(set(self.coordinate_pairs) - set([(player.x,player.y) for player in self.player_list]))
				possibleCoordinates = list(set(possibleCoordinates) - set([(food.x,food.y) for food in self.food_list]))
				food_positions = sample(range(len(possibleCoordinates)), 1)
				chosenCoordinate = possibleCoordinates[food_positions[0]]
				self.food_list.append(food.Food(chosenCoordinate[0],chosenCoordinate[1]))

		
		def setEpsilon(self,epsilon):
			for player in self.player_list:
				player.epsilon = epsilon

		def setListOfPlayers(self, listOfPlayers):
                        self.player_list = listOfPlayers

		def computeObservation(self, RGBMatrix, player):
			RGBRep = None
			if self.orientation == 3:
			# Facing left
				x_left = (player.x+(self.sight_radius)*2) - ((self.sight_sideways/2)*2)
				x_right = (player.x+ (self.sight_radius)*2) + ((self.sight_sideways/2)*2) + 2 
				y_up = (player.y+(self.sight_radius)*2) - (2*self.sight_radius)
				y_down = (player.y+(self.sight_radius)*2) + 2
				RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
				RGBRep = RGBRep.transpose((1,0,2))
				RGBRep = np.fliplr(RGBRep)
			elif self.orientation == 1:
				# Facing right
				x_left = (player.x+(self.sight_radius)*2) - ((self.sight_sideways/2)*2)
				x_right = (player.x+ (self.sight_radius)*2) + ((self.sight_sideways/2)*2) + 2 
				y_up = (player.y+(self.sight_radius)*2) + (2*self.sight_radius) + 2
				y_down = (player.y+(self.sight_radius)*2)
				RGBRep = RGBMatrix[x_left:x_right,y_down:y_up]
				RGBRep = RGBRep.transpose((1,0,2))
				RGBRep = RGBRep[::-1]
			elif self.orientation == 0:
				# Facing up
				x_left = (player.x+(self.sight_radius)*2) - (2*self.sight_radius)
				x_right = (player.x+ (self.sight_radius)*2) + 2
				y_up = (player.y+(self.sight_radius)*2) - ((self.sight_sideways/2)*2)
				y_down = (player.y+(self.sight_radius)*2) + ((self.sight_sideways/2)*2) + 2 
				RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
			elif self.orientation == 2:
				# Facing down
				x_left = (player.x+(self.sight_radius)*2)
				x_right = (player.x+ (self.sight_radius)*2) + (2*self.sight_radius) + 2
				y_up = (player.y+(self.sight_radius)*2) - ((self.sight_sideways/2)*2)
				y_down = (player.y+(self.sight_radius)*2) + ((self.sight_sideways/2)*2) + 2 
				RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
				RGBRep = np.fliplr(RGBRep)
				RGBRep = RGBRep[::-1]

			return RGBRep
        
		def computeFullObservation(self, RGBMatrix, player):
			RGBRep = None
			x_left = (self.sight_radius)*2
			x_right = -(self.sight_radius)*2 
			y_up = 2*self.sight_radius
			y_down = -2*self.sight_radius
			RGBRep = RGBMatrix[x_left:x_right,y_up:y_down]
			if self.orientation == 3:
				# Facing left
				RGBRep = RGBRep.transpose((1,0,2))
				RGBRep = np.fliplr(RGBRep)
			elif self.orientation == 1:
				# Facing right
				RGBRep = RGBRep.transpose((1,0,2))
				RGBRep = RGBRep[::-1]
			elif self.orientation == 0:
				# Facing up                
				RGBRep = RGBRep
			elif self.orientation == 2:
				# Facing down
				RGBRep = np.fliplr(RGBRep)
				RGBRep = RGBRep[::-1]

			return RGBRep
 
		def printGameStatistics(self):
			print("GAME PARAMETERS")
			print("Game Size X : "+str(self.x_size))
			print("Game Size Y : "+str(self.y_size))
			print("Number Of Players : "+str(self.num_players))
			print("Amount of Food : "+str(self.max_food_num))
			print("Interval Between Food Respawn : "+str(self.food_rate))
			print("Frames in one episode : "+str(self.perEpisodeFrame))
			if self.obsMode == "PARTIAL":
				print("Observation Size Y (PARTIAL MODE) : "+str(self.sight_radius))
				print("Observation Size X (PARTIAL MODE) : "+str(self.sight_sideways))
			else:
				print("Observation Size Y (PARTIAL MODE) : "+str(self.x_size))
				print("Observation Size X (PARTIAL MODE) : "+str(self.y_size))
