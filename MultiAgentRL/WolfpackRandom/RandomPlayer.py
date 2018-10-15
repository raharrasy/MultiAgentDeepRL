from random import randint
from player import Player

class RandomPlayer(Player) :
	def __init__(self,x,y,player_id,color = (0,255,0)):
		super(RandomPlayer, self).__init__(x,y,player_id,color)
	
	def act(self):
		self.action_counter += 1
		taken_action = randint(0,6)
		self.action_num = taken_action

		return self.action_num
