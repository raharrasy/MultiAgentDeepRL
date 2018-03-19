import pygame

class Food(object):
	def __init__(self,x,y,color = (255,0,0)):
		# Ensure that all inputs have been multiplied by 10 beforehand.
		self.x = x
		self.y = y
		self.color = color
