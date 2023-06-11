import pygame

class Piece:
	def __init__(self, piece_size, p_id):
		self.piece_size = piece_size
		self.p_id = p_id

		if self.p_id != 8:
			img_path = f'puzz-pieces/{self.p_id}.jpg'
			self.img = pygame.image.load(img_path)
			self.img = pygame.transform.scale(self.img, self.piece_size)
		else:
			self.img = None