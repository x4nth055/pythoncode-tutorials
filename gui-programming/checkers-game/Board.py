import pygame
from Tile import Tile
from Pawn import Pawn

class Board:
	def __init__(self,tile_width, tile_height, board_size):
		self.tile_width = tile_width
		self.tile_height = tile_height
		self.board_size = board_size
		self.selected_piece = None

		self.turn = "black"
		self.is_jump = False

		self.config = [
			['', 'bp', '', 'bp', '', 'bp', '', 'bp'],
			['bp', '', 'bp', '', 'bp', '', 'bp', ''],
			['', 'bp', '', 'bp', '', 'bp', '', 'bp'],
			['', '', '', '', '', '', '', ''],
			['', '', '', '', '', '', '', ''],
			['rp', '', 'rp', '', 'rp', '', 'rp', ''],
			['', 'rp', '', 'rp', '', 'rp', '', 'rp'],
			['rp', '', 'rp', '', 'rp', '', 'rp', '']
		]

		self.tile_list = self._generate_tiles()
		self._setup()

	def _generate_tiles(self):
		output = []
		for y in range(self.board_size):
			for x in range(self.board_size):
				output.append(
					Tile(x,  y, self.tile_width, self.tile_height)
				)
		return output

	def get_tile_from_pos(self, pos):
		for tile in self.tile_list:
			if (tile.x, tile.y) == (pos[0], pos[1]):
				return tile

	def _setup(self):
		for y_ind, row in enumerate(self.config):
			for x_ind, x in enumerate(row):
				tile = self.get_tile_from_pos((x_ind, y_ind))
				if x != '':
					if x[-1] == 'p':
						color = 'red' if x[0] == 'r' else 'black'
						tile.occupying_piece = Pawn(x_ind, y_ind, color, self)

	def handle_click(self, pos):
		x, y = pos[0], pos[-1]
		if x >= self.board_size or y >= self.board_size:
			x = x // self.tile_width
			y = y // self.tile_height
		clicked_tile = self.get_tile_from_pos((x, y))

		if self.selected_piece is None:
			if clicked_tile.occupying_piece is not None:
				if clicked_tile.occupying_piece.color == self.turn:
					self.selected_piece = clicked_tile.occupying_piece
		elif self.selected_piece._move(clicked_tile):
			if not self.is_jump:
				self.turn = 'red' if self.turn == 'black' else 'black'
			else:
				if len(clicked_tile.occupying_piece.valid_jumps()) == 0:
					self.turn = 'red' if self.turn == 'black' else 'black'
		elif clicked_tile.occupying_piece is not None:
			if clicked_tile.occupying_piece.color == self.turn:
				self.selected_piece = clicked_tile.occupying_piece

	def draw(self, display):
		if self.selected_piece is not None:
			self.get_tile_from_pos(self.selected_piece.pos).highlight = True
			if not self.is_jump:
				for tile in self.selected_piece.valid_moves():
					tile.highlight = True
			else:
				for tile in self.selected_piece.valid_jumps():
					tile[0].highlight = True

		for tile in self.tile_list:
			tile.draw(display)