import pygame

class Tile:
	def __init__(self, x, y, tile_width, tile_height):
		self.x = x
		self.y = y
		self.pos = (x, y)
		self.tile_width = tile_width
		self.tile_height = tile_height
		self.abs_x = x * tile_width
		self.abs_y = y * tile_height
		self.abs_pos = (self.abs_x, self.abs_y)

		self.color = 'light' if (x + y) % 2 == 0 else 'dark'
		self.draw_color = (220, 189, 194) if self.color == 'light' else (53, 53, 53)
		self.highlight_color = (100, 249, 83) if self.color == 'light' else (0, 228, 10)

		self.occupying_piece = None
		self.coord = self.get_coord()
		self.highlight = False
		self.rect = pygame.Rect(
			self.abs_x,
			self.abs_y,
			self.tile_width,
			self.tile_height
		)

	def get_coord(self):
		columns = 'abcdefgh'
		return columns[self.x] + str(self.y + 1)

	def draw(self, display):
		if self.highlight:
			pygame.draw.rect(display, self.highlight_color, self.rect)
		else:
			pygame.draw.rect(display, self.draw_color, self.rect)

		if self.occupying_piece != None:
			centering_rect = self.occupying_piece.img.get_rect()
			centering_rect.center = self.rect.center
			display.blit(self.occupying_piece.img, centering_rect.topleft)