import pygame

from settings import PLAYER_SPEED, BULLET_SIZE
from bullet import Bullet

class Ship(pygame.sprite.Sprite):
	def __init__(self, pos, size):
		super().__init__()
		self.x = pos[0]
		self.y = pos[1]

		# ship info 
		img_path = 'assets/ship/ship.png'
		self.image = pygame.image.load(img_path)
		self.image = pygame.transform.scale(self.image, (size, size))
		self.rect = self.image.get_rect(topleft = pos)
		self.mask = pygame.mask.from_surface(self.image)
		self.ship_speed = PLAYER_SPEED

		# ship status
		self.life = 3
		self.player_bullets = pygame.sprite.Group()


	def move_left(self):
		self.rect.x -= self.ship_speed

	def move_up(self):
		self.rect.y -= self.ship_speed

	def move_right(self):
		self.rect.x += self.ship_speed

	def move_bottom(self):
		self.rect.y += self.ship_speed

	def _shoot(self):
		specific_pos = (self.rect.centerx - (BULLET_SIZE // 2), self.rect.y)
		self.player_bullets.add(Bullet(specific_pos, BULLET_SIZE, "player"))

	def update(self):
		self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))