import pygame
from cell import Cell

class Maze:
	def __init__(self, cols, rows):
		self.cols = cols
		self.rows = rows
		self.thickness = 4
		self.grid_cells = [Cell(col, row, self.thickness) for row in range(self.rows) for col in range(self.cols)]

	# carve grid cell walls
	def remove_walls(self, current, next):
		dx = current.x - next.x
		if dx == 1:
			current.walls['left'] = False
			next.walls['right'] = False
		elif dx == -1:
			current.walls['right'] = False
			next.walls['left'] = False
		dy = current.y - next.y
		if dy == 1:
			current.walls['top'] = False
			next.walls['bottom'] = False
		elif dy == -1:
			current.walls['bottom'] = False
			next.walls['top'] = False

	# generates maze
	def generate_maze(self):
		current_cell = self.grid_cells[0]
		array = []
		break_count = 1
		while break_count != len(self.grid_cells):
			current_cell.visited = True
			next_cell = current_cell.check_neighbors(self.cols, self.rows, self.grid_cells)
			if next_cell:
				next_cell.visited = True
				break_count += 1
				array.append(current_cell)
				self.remove_walls(current_cell, next_cell)
				current_cell = next_cell
			elif array:
				current_cell = array.pop()
		return self.grid_cells
