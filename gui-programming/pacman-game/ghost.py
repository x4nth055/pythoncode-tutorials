import pygame
import random
import time

from settings import WIDTH, CHAR_SIZE, GHOST_SPEED

class Ghost(pygame.sprite.Sprite):
	def __init__(self, row, col, color):
		super().__init__()
		self.abs_x = (row * CHAR_SIZE)
		self.abs_y = (col * CHAR_SIZE)

		self.rect = pygame.Rect(self.abs_x, self.abs_y, CHAR_SIZE, CHAR_SIZE)
		self.move_speed = GHOST_SPEED
		self.color = pygame.Color(color)
		self.move_directions = [(-1,0), (0,-1), (1,0), (0,1)]

		self.moving_dir = "up"
		self.img_path = f'assets/ghosts/{color}/'
		self.img_name = f'{self.moving_dir}.png'
		self.image = pygame.image.load(self.img_path + self.img_name)
		self.image = pygame.transform.scale(self.image, (CHAR_SIZE, CHAR_SIZE))
		self.rect = self.image.get_rect(topleft = (self.abs_x, self.abs_y))
		self.mask = pygame.mask.from_surface(self.image)

		self.directions = {'left': (-self.move_speed, 0), 'right': (self.move_speed, 0), 'up': (0, -self.move_speed), 'down': (0, self.move_speed)}
		self.keys = ['left', 'right', 'up', 'down']
		self.direction = (0, 0)

	def move_to_start_pos(self):
		self.rect.x = self.abs_x
		self.rect.y = self.abs_y

	def is_collide(self, x, y, walls_collide_list):
		tmp_rect = self.rect.move(x, y)
		if tmp_rect.collidelist(walls_collide_list) == -1:
			return False
		return True

	def _animate(self):
		self.img_name = f'{self.moving_dir}.png'
		self.image = pygame.image.load(self.img_path + self.img_name)
		self.image = pygame.transform.scale(self.image, (CHAR_SIZE, CHAR_SIZE))
		self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))

	def update(self, walls_collide_list):
		# ghost movement
		available_moves = []
		for key in self.keys:
			if not self.is_collide(*self.directions[key], walls_collide_list):
				available_moves.append(key)
		
		randomizing = False if len(available_moves) <= 2 and self.direction != (0,0) else True
		# 60% chance of randomizing ghost move
		if randomizing and random.randrange( 0,100 ) <= 60:
			self.moving_dir = random.choice(available_moves)
			self.direction = self.directions[self.moving_dir]

		if not self.is_collide(*self.direction, walls_collide_list):
			self.rect.move_ip(self.direction)
		else:
			self.direction = (0,0)

		# teleporting to the other side of the map
		if self.rect.right <= 0:
			self.rect.x = WIDTH
		elif self.rect.left >= WIDTH:
			self.rect.x = 0

		self._animate()
