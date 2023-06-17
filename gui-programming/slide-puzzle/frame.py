import pygame
import random

from cell import Cell
from piece import Piece

class Frame:
	def __init__(self, frame_size):
		self.grid_size = 3
		self.cell_width = frame_size // self.grid_size
		self.cell_height = frame_size // self.grid_size
		self.cell_size = (self.cell_width, self.cell_height)
		
		self.grid = self._generate_cell()
		self.pieces = self._generate_piece()

		self._setup()
		self.randomize_puzzle()

	def _generate_cell(self):
		cells = []
		c_id = 0
		for col in range(self.grid_size):
			new_row = []
			for row in range(self.grid_size):
				new_row.append(Cell(row, col, self.cell_size, c_id))
				c_id += 1
			cells.append(new_row)
		return cells

	def _generate_piece(self):
		puzzle_pieces = []
		p_id = 0
		for col in range(self.grid_size):
			for row in range(self.grid_size):
				puzzle_pieces.append(Piece(self.cell_size, p_id))
				p_id += 1
		return puzzle_pieces

	def _setup(self):
		for row in self.grid:
			for cell in row:
				tile_piece = self.pieces[-1]
				cell.occupying_piece = tile_piece
				self.pieces.remove(tile_piece)

	def randomize_puzzle(self):
		moves = [(0, 1),(0, -1),(1, 0),(-1, 0)]
		for i in range(30):
			shuffle_move = random.choice(moves)
			for row in self.grid:
				for cell in row:
					tile_x = self.grid.index(row) + shuffle_move[0]
					tile_y = row.index(cell) + shuffle_move[1]
					if tile_x >= 0 and tile_x <= 2 and tile_y >= 0 and tile_y <= 2:
						new_cell = self.grid[tile_x][tile_y]
						if new_cell.occupying_piece.img == None:
							c = (cell, new_cell)
							try:
								c[0].occupying_piece, c[1].occupying_piece = c[1].occupying_piece, c[0].occupying_piece
							except:
								return False
					else:
						continue

	def _is_move_valid(self, click):
		moves = {
			79: (0, 1),
			80: (0, -1),
			81: (1, 0),
			82: (-1, 0)
		}
		for row in self.grid:
			for cell in row:
				move = moves[click.scancode]
				tile_x = self.grid.index(row) + move[0]
				tile_y = row.index(cell) + move[1]
				if tile_x >= 0 and tile_x <= 2 and tile_y >= 0 and tile_y <= 2:
					new_cell = self.grid[tile_x][tile_y]
					if new_cell.occupying_piece.img == None:
						return (cell, new_cell)
				else:
					continue

	def handle_click(self, click):
		c = self._is_move_valid(click)
		try:
			c[0].occupying_piece, c[1].occupying_piece = c[1].occupying_piece, c[0].occupying_piece
		except:
			return False

	def draw(self, display):
		for row in self.grid:
			for cell in row: 
				cell.draw(display)
