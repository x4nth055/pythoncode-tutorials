import pygame

class Cell:
	def __init__(self, row, col, cell_size, c_id):
		self.row = row
		self.col = col
		self.cell_size = cell_size
		self.width = self.cell_size[0]
		self.height = self.cell_size[1]
		self.abs_x = row * self.width
		self.abs_y = col * self.height

		self.c_id = c_id

		self.rect = pygame.Rect(
			self.abs_x,
			self.abs_y,
			self.width,
			self.height
		)

		self.occupying_piece = None

	def draw(self, display):
		pygame.draw.rect(display, (0,0,0), self.rect)
		if self.occupying_piece != None and self.occupying_piece.p_id != 8:
			centering_rect = self.occupying_piece.img.get_rect()
			centering_rect.center = self.rect.center
			display.blit(self.occupying_piece.img, centering_rect.topleft)