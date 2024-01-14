import pygame
import math
from cell import Cell
from sudoku import Sudoku
from clock import Clock

from settings import WIDTH, HEIGHT, N_CELLS, CELL_SIZE

pygame.font.init()

class Table:
	def __init__(self, screen):
		self.screen = screen

		self.puzzle = Sudoku(N_CELLS, (N_CELLS * N_CELLS) // 2)
		self.clock = Clock()

		self.answers = self.puzzle.puzzle_answers()
		self.answerable_table = self.puzzle.puzzle_table()
		self.SRN = self.puzzle.SRN

		self.table_cells = []
		self.num_choices = []
		self.clicked_cell = None
		self.clicked_num_below = None
		self.cell_to_empty = None
		self.making_move = False
		self.guess_mode = True

		self.lives = 3
		self.game_over = False

		self.delete_button = pygame.Rect(0, (HEIGHT + CELL_SIZE[1]), (CELL_SIZE[0] * 3), (CELL_SIZE[1]))
		self.guess_button = pygame.Rect((CELL_SIZE[0] * 6), (HEIGHT + CELL_SIZE[1]), (CELL_SIZE[0] * 3), (CELL_SIZE[1]))
		self.font = pygame.font.SysFont('Bauhaus 93', (CELL_SIZE[0] // 2))
		self.font_color = pygame.Color("white")
	
		self._generate_game()
		self.clock.start_timer()


	def _generate_game(self):
		# generating sudoku table
		for y in range(N_CELLS):
			for x in range(N_CELLS):
				cell_value = self.answerable_table[y][x]
				is_correct_guess = True if cell_value != 0 else False
				self.table_cells.append(Cell(x, y, CELL_SIZE, cell_value, is_correct_guess))

		# generating number choices
		for x in range(N_CELLS):
			self.num_choices.append(Cell(x, N_CELLS, CELL_SIZE, x + 1))


	def _draw_grid(self):
		grid_color = (50, 80, 80)
		pygame.draw.rect(self.screen, grid_color, (-3, -3, WIDTH + 6, HEIGHT + 6), 6)

		i = 1
		while (i * CELL_SIZE[0]) < WIDTH:
			line_size = 2 if i % 3 > 0 else 4
			pygame.draw.line(self.screen, grid_color, ((i * CELL_SIZE[0]) - (line_size // 2), 0), ((i * CELL_SIZE[0]) - (line_size // 2), HEIGHT), line_size)
			pygame.draw.line(self.screen, grid_color, (0, (i * CELL_SIZE[0]) - (line_size // 2)), (HEIGHT, (i * CELL_SIZE[0]) - (line_size // 2)), line_size)
			i += 1


	def _draw_buttons(self):
		# adding delete button details
		dl_button_color = pygame.Color("red")
		pygame.draw.rect(self.screen, dl_button_color, self.delete_button)
		del_msg = self.font.render("Delete", True, self.font_color)
		self.screen.blit(del_msg, (self.delete_button.x + (CELL_SIZE[0] // 2), self.delete_button.y + (CELL_SIZE[1] // 4)))
		# adding guess button details
		gss_button_color = pygame.Color("blue") if self.guess_mode else pygame.Color("purple")
		pygame.draw.rect(self.screen, gss_button_color, self.guess_button)
		gss_msg = self.font.render("Guess: On" if self.guess_mode else "Guess: Off", True, self.font_color)
		self.screen.blit(gss_msg, (self.guess_button.x + (CELL_SIZE[0] // 3), self.guess_button.y + (CELL_SIZE[1] // 4)))


	def _get_cell_from_pos(self, pos):
		for cell in self.table_cells:
			if (cell.row, cell.col) == (pos[0], pos[1]):
				return cell


	# checking rows, cols, and subgroups for adding guesses on each cell
	def _not_in_row(self, row, num):
		for cell in self.table_cells:
			if cell.row == row:
				if cell.value == num:
					return False
		return True
	
	def _not_in_col(self, col, num):
		for cell in self.table_cells:
			if cell.col == col:
				if cell.value == num:
					return False
		return True

	def _not_in_subgroup(self, rowstart, colstart, num):
		for x in range(self.SRN):
			for y in range(self.SRN):
				current_cell = self._get_cell_from_pos((rowstart + x, colstart + y))
				if current_cell.value == num:
					return False
		return True


	# remove numbers in guess if number already guessed in the same row, col, subgroup correctly
	def _remove_guessed_num(self, row, col, rowstart, colstart, num):
		for cell in self.table_cells:
			if cell.row == row and cell.guesses != None:
				for x_idx,guess_row_val in enumerate(cell.guesses):
					if guess_row_val == num:
						cell.guesses[x_idx] = 0
			if cell.col == col and cell.guesses != None:
				for y_idx,guess_col_val in enumerate(cell.guesses):
					if guess_col_val == num:
						cell.guesses[y_idx] = 0

		for x in range(self.SRN):
			for y in range(self.SRN):
				current_cell = self._get_cell_from_pos((rowstart + x, colstart + y))
				if current_cell.guesses != None:
					for idx,guess_val in enumerate(current_cell.guesses):
						if guess_val == num:
							current_cell.guesses[idx] = 0


	def handle_mouse_click(self, pos):
		x, y = pos[0], pos[1]

		# getting table cell clicked
		if x <= WIDTH and y <= HEIGHT:
			x = x // CELL_SIZE[0]
			y = y // CELL_SIZE[1]
			clicked_cell = self._get_cell_from_pos((x, y))

			# if clicked empty cell
			if clicked_cell.value == 0:
				self.clicked_cell = clicked_cell
				self.making_move = True

			# clicked unempty cell but with wrong number guess
			elif clicked_cell.value != 0 and clicked_cell.value != self.answers[y][x]:
				self.cell_to_empty = clicked_cell

		# getting number selected
		elif x <= WIDTH and y >= HEIGHT and y <= (HEIGHT + CELL_SIZE[1]):
			x = x // CELL_SIZE[0]
			self.clicked_num_below = self.num_choices[x].value

		# deleting numbers
		elif x <= (CELL_SIZE[0] * 3) and y >= (HEIGHT + CELL_SIZE[1]) and y <= (HEIGHT + CELL_SIZE[1] * 2):
			if self.cell_to_empty:
				self.cell_to_empty.value = 0
				self.cell_to_empty = None

		# selecting modes
		elif x >= (CELL_SIZE[0] * 6) and y >= (HEIGHT + CELL_SIZE[1]) and y <= (HEIGHT + CELL_SIZE[1] * 2):
			self.guess_mode = True if not self.guess_mode else False

		# if making a move
		if self.clicked_num_below and self.clicked_cell != None and self.clicked_cell.value == 0:
			current_row = self.clicked_cell.row
			current_col = self.clicked_cell.col
			rowstart = self.clicked_cell.row - self.clicked_cell.row % self.SRN
			colstart = self.clicked_cell.col - self.clicked_cell.col % self.SRN

			if self.guess_mode:
				# checking the vertical group, the horizontal group, and the subgroup
				if self._not_in_row(current_row, self.clicked_num_below) and self._not_in_col(current_col, self.clicked_num_below):
					if self._not_in_subgroup(rowstart, colstart, self.clicked_num_below):
						if self.clicked_cell.guesses != None:
							self.clicked_cell.guesses[self.clicked_num_below - 1] = self.clicked_num_below
			else:
				self.clicked_cell.value = self.clicked_num_below
				# if the player guess correctly
				if self.clicked_num_below == self.answers[self.clicked_cell.col][self.clicked_cell.row]:
					self.clicked_cell.is_correct_guess = True
					self.clicked_cell.guesses = None
					self._remove_guessed_num(current_row, current_col, rowstart, colstart, self.clicked_num_below)
				# if guess is wrong
				else:
					self.clicked_cell.is_correct_guess = False
					self.clicked_cell.guesses = [0 for x in range(9)]
					self.lives -= 1
			self.clicked_num_below = None
			self.making_move = False
		else:
			self.clicked_num_below = None


	def _puzzle_solved(self):
		check = None
		for cell in self.table_cells:
			if cell.value == self.answers[cell.col][cell.row]:
				check = True
			else:
				check = False
				break
		return check


	def update(self):
		[cell.update(self.screen, self.SRN) for cell in self.table_cells]

		[num.update(self.screen) for num in self.num_choices]

		self._draw_grid()
		self._draw_buttons()

		if self._puzzle_solved() or self.lives == 0:
			self.clock.stop_timer()
			self.game_over = True
		else:
			self.clock.update_timer()

		self.screen.blit(self.clock.display_timer(), (WIDTH // self.SRN,HEIGHT + CELL_SIZE[1]))