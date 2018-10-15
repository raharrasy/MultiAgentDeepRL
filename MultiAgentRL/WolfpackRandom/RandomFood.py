from random import randint
from food import Food 

class RandomFood(Food):
	def __init__(self,x=0,y=0,color = (255,0,0), dead_period = 15):
		# Ensure that all inputs have been multiplied by 10 beforehand.
		super(RandomFood, self).__init__(x,y,color = (255,0,0), dead_period = 15)
		#self.filename = filename

	def act(self):
		if not self.is_dead:
			self.action_counter += 1
		taken_action = randint(0,6)

		if self.is_dead:
			taken_action = 4
		self.action_num = taken_action

		return self.action_num