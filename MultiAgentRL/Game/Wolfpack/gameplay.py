from random import randint
from random import sample
from player import Player
from food import Food
import numpy as np

class State(object) :
		def __init__(self, sight_sideways, sight_radius,num_players, max_food_num, arena_size, levelMap, food_rate = 5, num_game_frames = 100000,coop_radius = 12, groupMultiplier = 2, obsMode = "PARTIAL"):
			self.obsMode = obsMode
			self.x_size = arena_size[0]
			self.y_size = arena_size[1]
			self.max_food_num = max_food_num
			self.coordinate_pairs = [(i*2,j*2) for i in range(1,arena_size[0]-1) for j in range(1,arena_size[1]-1)]
			self.obstacleCoord = [(iy*2,ix*2) for ix, row in enumerate(levelMap) for iy, i in enumerate(row) if i]
			self.possibleCoordinates = list(set(self.coordinate_pairs) - set(self.obstacleCoord))
			random_player_positions = sample(range(len(self.possibleCoordinates)), num_players)
			# coordinates = [self.possibleCoordinates[ii] for ii in random_player_positions]
			# self.player_list = [Player(i,j) for (i,j) in coordinates]
			self.player_list = []
			self.food_list = []
			#Find out about this
			self.food_rate = food_rate
			#Find out about this
			self.waiting_time = food_rate
			self.remaining_game_frames = num_game_frames
			self.sight_sideways = sight_sideways
			self.sight_radius = sight_radius
			self.coopRadius = coop_radius
			self.wolfPerCapture = []
			self.groupMultiplier = groupMultiplier
			self.addFood()
		
		def reset(self,num_players,levelMap, food_rate = 5, num_game_frames = 100000):
			self.coordinate_pairs = [(i*2,j*2) for i in range(1,self.x_size-1) for j in range(1,self.y_size-1)]
			self.obstacleCoord = [(iy*2,ix*2) for ix, row in enumerate(levelMap) for iy, i in enumerate(row) if i]
			self.possibleCoordinates = list(set(self.coordinate_pairs) - set(self.obstacleCoord))
			random_player_positions = sample(range(len(self.possibleCoordinates)), num_players)
			coordinates = [self.possibleCoordinates[ii] for ii in random_player_positions]
			ii = 0
			for player in self.player_list:
				player.reset(coordinates[ii])
				ii += 1

			ii = 0
			foodCoordinates = list(set(self.possibleCoordinates) - set(coordinates))
			random_food_positions = sample(range(len(foodCoordinates)), len(self.food_list))
			food_coordinates = [foodCoordinates[jj] for jj in random_food_positions]
			for food in self.food_list:
				food.reset(food_coordinates[ii])
				ii += 1

			self.food_rate = food_rate
			self.waiting_time = food_rate
			self.remaining_game_frames = num_game_frames
			self.wolfPerCapture = []
			self.addFood()

		def updateState(self):
			
			collectiveAction,collectiveActionFood = self.getCollectiveAct()
			self.update_state(collectiveAction,collectiveActionFood)
		
		def sense(self,newRGB,ExperienceFlag=False):
			self.RGBMatrix = newRGB
			RGBObservation = None
			for player in self.player_list:
				if self.obsMode == "PARTIAL":
					RGBObservation = self.computeObservation(self.RGBMatrix, player)
				else:
					RGBObservation = self.computeFullObservation(self.RGBMatrix, player)
				player.sense(RGBObservation,ExperienceFlag)

			for food in self.food_list:
				if self.obsMode == "PARTIAL":
					RGBObservation = self.computeObservation(self.RGBMatrix, food)
				else:
					RGBObservation = self.computeFullObservation(self.RGBMatrix, food)
				food.sense(RGBObservation,ExperienceFlag)

		def learn(self):
			for player in self.player_list:
				player.learn()

			for food in self.food_list:
				food.learn()

		
		def getCollectiveAct(self):
			collectiveAct = [player.act() for player in self.player_list]
			collectiveActFood = [player.act() for player in self.food_list]
			return collectiveAct,collectiveActFood

		def revive(self):
			#find possible locations to revive dead player
			possible_coordinates = set(self.possibleCoordinates)-set([player.getIndex() for player in self.player_list])
			possible_coordinates = possible_coordinates - set([food.getIndex() for food in self.food_list])
			for food in self.food_list:
				if food.remaining_time == 0 and food.is_dead:
					idx = sample(range(len(list(possible_coordinates))), 1)
					coord = list(possible_coordinates)[idx[0]]
					possible_coordinates = possible_coordinates - set([coord])
					food.setAlive()
					food.x = coord[0]
					food.y = coord[1]
		
		def update_state(self,collectiveAct,collectiveActFood):
			self.waiting_time -= 1
			self.remaining_game_frames -= 1
			self.update_status()
			self.revive()


			prev_player_position = [(player.x,player.y) for player in self.player_list]
			prev_player_orientation = [player.orientation for player in self.player_list] 
			prev_food_position = [(player.x,player.y) for player in self.food_list]
			prev_food_orientation = [player.orientation for player in self.food_list]

			#Calculate new player positions
			update_results = self.calculate_new_position(collectiveAct,prev_player_position,prev_player_orientation)
			post_player_position = [(a,b) for (a,b,c) in update_results]
			post_player_orientation = [c for (a,b,c) in update_results]
			for ii in range(len(self.player_list)):
				self.player_list[ii].orientation = post_player_orientation[ii]
			
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

			#Calculate new food locations
			update_results = self.calculate_new_position(collectiveActFood,prev_food_position,prev_food_orientation)
			post_food_position = [(a,b) for (a,b,c) in update_results]
			post_food_orientation = [c for (a,b,c) in update_results]
			for ii in range(len(self.food_list)):
				self.food_list[ii].orientation = post_food_orientation[ii]
			
			#Calculate food intersection
			a, seen, result = post_food_position, set(), {}
			for idx, item in enumerate(a):
				if not self.food_list[idx].is_dead:
					if item not in seen:
						result[item] = [idx]
						seen.add(item)
					else:
						result[item].append(idx)

			groupings = list(result.values())
			doubles = [t for t in groupings if len(t) >= 2]
			res = set([item for sublist in doubles for item in sublist])
			for ii in range(len(self.food_list)):
				if ii not in res:
					self.food_list[ii].setIndex(post_food_position[ii])
				else:
					self.food_list[ii].setIndex(prev_food_position[ii])
						
			# Calculate player points and food status
			self.update_food_status(self.player_list)
			
			
			if self.waiting_time == 0:
				self.waiting_time = self.food_rate
				self.addFood()

		def calculate_new_position(self, collectiveAct, prev_player_position,prev_player_orientation):
			zipped_data = list(zip(collectiveAct, prev_player_position,prev_player_orientation))
			result = [self.calculate_indiv_position(a,(b,c),d) for (a,(b,c),d) in zipped_data]
			return result
		
		def calculate_indiv_position(self,action,pair,orientation):
			x = pair[0]
			y = pair[1]
			next_x = x
			next_y = y

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

				if (next_x,next_y) in set(self.possibleCoordinates):
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

				if (next_x,next_y) in set(self.possibleCoordinates):
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

				if (next_x,next_y) in set(self.possibleCoordinates):
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

				if (next_x,next_y) in set(self.possibleCoordinates):
				 	return (next_x,next_y,orientation)
				else:
				 	return (x,y,orientation)
			#stay still
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
			else:
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

		def update_food_status(self,player_list):
			for food in self.food_list:
				food.add_point(0)

			enumFood = list(enumerate(self.food_list))
			food_locations = [(food.x,food.y) for idx,food in enumFood if not food.isDead()]
			food_id = [idx for idx,food in enumFood if not food.isDead()]

			player_locations = [(player.x,player.y) for player in self.player_list]
			set_of_food_location = set(food_locations)
			rewardListPlayer = [0]*len(player_locations)

			for player in player_list:
				if (player.getIndex() in set_of_food_location):
					center = player.getIndex()
					center = (center[0]+1, center[1]+1)
					enumerated = enumerate(player_locations)
					close = [x for (x,(a,b)) in enumerated if abs(a-center[0]) + abs(b-center[1]) <= self.coopRadius]
					for x in close:
						if len(close) < 2:
							rewardListPlayer[x] += len(close)/4.0
						else:
							rewardListPlayer[x] += self.groupMultiplier*len(close)/4.0
					self.wolfPerCapture.append(len(close))
					food_index = food_locations.index(player.getIndex())
					self.food_list[food_id[food_index]].add_point(-1/4.0)
					self.food_list[food_id[food_index]].setDead()

			for ii in range(len(rewardListPlayer)):
				self.player_list[ii].add_player_point(rewardListPlayer[ii])

		def update_status(self):
			for food in self.food_list:
				if food.isDead():
					food.remaining_time -= 1
		
		def addFood(self):
			while len(self.food_list) < self.max_food_num:
				possibleCoordinates = list(set(self.possibleCoordinates) - set([(player.x,player.y) for player in self.player_list]))
				possibleCoordinates = list(set(possibleCoordinates) - set([(food.x,food.y) for food in self.food_list]))
				food_positions = sample(range(len(possibleCoordinates)), 1)
				chosenCoordinate = possibleCoordinates[food_positions[0]]
				self.food_list.append(Food(chosenCoordinate[0],chosenCoordinate[1]))
				self.food_list[-1].NN.sess.graph.finalize()


		def setEpsilon(self,epsilon):
			for player in self.player_list:
				player.epsilon = epsilon
			for food in self.food_list:
				food.epsilon = epsilon


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