import pygame
import random
import numpy as np
from random import randint
from random import random

class Player(object) :
	def __init__(self,x,y,player_id,color = (0,255,0)):
		self.x = x
		self.y = y
		self.id = player_id
		self.color = color		
		self.action_num = 0
		self.point = 0
		self.orientation = 0
		self.action_counter = 0
		self.playerLastPoint = self.point

	def reset(self,location):
		self.x = location[0]
		self.y = location[1]		
		self.action_num = 0
		self.point = 0
		self.orientation = 0

	def add_player_point(self, point):
		self.point += point
		self.playerLastPoint = point
	
	def setIndex(self,newPosition):
		self.x = newPosition[0]
		self.y = newPosition[1]
	
	def getIndex(self):
		return (self.x,self.y)
	
	def act(self):
		pass

	def sense(self,RGBMatrix, ExperienceFlag=False, LastExpFlag=False):
		pass


	def learn(self):
		pass

	def checkpointing(self,step):
		pass
