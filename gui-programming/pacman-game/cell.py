import pygame

class Cell(pygame.sprite.Sprite):
	def __init__(self, row, col, length, width):
		super().__init__()
		self.width = length
		self.height = width
		self.id = (row, col)
		self.abs_x = row * self.width
		self.abs_y = col * self.height

		self.rect = pygame.Rect(self.abs_x,self.abs_y,self.width,self.height)

		self.occupying_piece = None

	def update(self, screen):
		pygame.draw.rect(screen, pygame.Color("blue2"), self.rect)