import pygame, sys
import random
from settings import WIDTH, HEIGHT

pygame.init()

class Ball:
	def __init__(self, x, y, radius):
		self.x = x
		self.y = y
		self.radius = radius
		self.rect = pygame.Rect(self.x, self.y, radius, radius)
		self.color = pygame.Color("red")
		self.direction = None
		self.speed_x = 0
		self.speed_y = 0
		self._random_direction()

	def _random_direction(self):
		direction = ("right", "left")
		self.direction = random.choice(direction)

	def _ball_movement(self):
		# horizontal handling
		if self.direction == "right":
			self.speed_x = 18
		else:
			self.speed_x = -18

		# vertical handling
		if self.rect.y >= HEIGHT - self.radius:
			self.speed_y = -18
		elif self.rect.y <= 0 + self.radius:
			self.speed_y = 18

		# wall bounce handling
		self.rect.x += self.speed_x
		self.rect.y += self.speed_y

	def update(self, screen):
		self._ball_movement()
		pygame.draw.rect(screen, self.color, self.rect)