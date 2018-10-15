import pygame
import random
from random import randint
from random import random
import numpy as np

class Food(object):
	def __init__(self,x,y,color = (255,0,0), dead_period = 15):
		# Ensure that all inputs have been multiplied by 10 beforehand.
		self.x = x
		self.y = y
		self.color = color
		self.action_num = 0
		self.orientation = 0
		self.is_dead = False
		self.dead_period = dead_period
		self.remaining_time = 0
		self.point = 0
		self.action_counter = 0
		#self.filename = filename

	def reset(self,location):
		self.x = location[0]
		self.y = location[1]		
		self.action_num = 0
		self.point = 0
		self.remaining_time = 0
		self.is_dead = False
		self.orientation = 0

	def add_point(self, point):
		pass
	
	def setIndex(self,newPosition):
		self.x = newPosition[0]
		self.y = newPosition[1]
	
	def getIndex(self):
		return (self.x,self.y)

	def setAlive(self):
		self.is_dead = False
	
	def setDead(self):
		self.is_dead = True
		self.remaining_time = self.dead_period
	
	def getRemainingTime(self):
		return self.remaining_time
	
	def isDead(self):
		return self.is_dead

	def act(self):
		pass

	def sense(self,RGBMatrix,ExperienceFlag=False, LastExpFlag=False):		
		pass

	def learn(self):
		pass

	def save(self,filename):
		pass


	def checkpointing(self, filename):
		pass

	def checkpointing2(self, filename, step=0):
		pass
