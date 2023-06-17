import pygame

pygame.font.init()

class Game:
	def __init__(self):
		self.font = pygame.font.SysFont("Courier New", 35)
		self.background_color = (255, 174, 66)
		self.message_color = (17, 53, 165)

	def arrow_key_clicked(self, click):
		try:
			if click.key == pygame.K_LEFT or click.key == pygame.K_RIGHT or click.key == pygame.K_UP or click.key == pygame.K_DOWN:
				return(True)
		except:
			return(False)

	def is_game_over(self, frame):
		for row in frame.grid:
			for cell in row:
				piece_id = cell.occupying_piece.p_id
				if cell.c_id == piece_id:
					is_arranged = True
				else:
					is_arranged = False
					break
		return is_arranged

	def message(self, screen):
		screen.fill(self.background_color, (5, 460, 440, 35))
		instructions = self.font.render('You Win!!', True, self.message_color)
		screen.blit(instructions,(125,460))