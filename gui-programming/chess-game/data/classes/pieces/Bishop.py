import pygame

from data.classes.Piece import Piece

class Bishop(Piece):
	def __init__(self, pos, color, board):
		super().__init__(pos, color, board)

		img_path = 'data/imgs/' + color[0] + '_bishop.png'
		self.img = pygame.image.load(img_path)
		self.img = pygame.transform.scale(self.img, (board.tile_width - 20, board.tile_height - 20))

		self.notation = 'B'


	def get_possible_moves(self, board):
		output = []

		moves_ne = []
		for i in range(1, 8):
			if self.x + i > 7 or self.y - i < 0:
				break
			moves_ne.append(board.get_square_from_pos(
				(self.x + i, self.y - i)
			))
		output.append(moves_ne)

		moves_se = []
		for i in range(1, 8):
			if self.x + i > 7 or self.y + i > 7:
				break
			moves_se.append(board.get_square_from_pos(
				(self.x + i, self.y + i)
			))
		output.append(moves_se)

		moves_sw = []
		for i in range(1, 8):
			if self.x - i < 0 or self.y + i > 7:
				break
			moves_sw.append(board.get_square_from_pos(
				(self.x - i, self.y + i)
			))
		output.append(moves_sw)

		moves_nw = []
		for i in range(1, 8):
			if self.x - i < 0 or self.y - i < 0:
				break
			moves_nw.append(board.get_square_from_pos(
				(self.x - i, self.y - i)
			))
		output.append(moves_nw)

		return output
