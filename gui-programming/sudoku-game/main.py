import pygame, sys
from settings import WIDTH, HEIGHT, CELL_SIZE
from table import Table

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + (CELL_SIZE[1] * 3)))
pygame.display.set_caption("Sudoku")

pygame.font.init()

class Main:
	def __init__(self, screen):
		self.screen = screen
		self.FPS = pygame.time.Clock()
		self.lives_font = pygame.font.SysFont("monospace", CELL_SIZE[0] // 2)
		self.message_font = pygame.font.SysFont('Bauhaus 93', (CELL_SIZE[0]))
		self.color = pygame.Color("darkgreen")

	def main(self):
		table = Table(self.screen)

		while True:
			self.screen.fill("gray")
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
				if event.type == pygame.MOUSEBUTTONDOWN:
					if not table.game_over:
						table.handle_mouse_click(event.pos)

			# lower screen display
			if not table.game_over:
				my_lives = self.lives_font.render(f"Lives Left: {table.lives}", True, pygame.Color("black"))
				self.screen.blit(my_lives, ((WIDTH // table.SRN) - (CELL_SIZE[0] // 2), HEIGHT + (CELL_SIZE[1] * 2.2)))

			else:
				if table.lives <= 0:
					message = self.message_font.render("GAME OVER!!", True, pygame.Color("red"))
					self.screen.blit(message, (CELL_SIZE[0] + (CELL_SIZE[0] // 2), HEIGHT + (CELL_SIZE[1] * 2)))
				elif table.lives > 0:
					message = self.message_font.render("You Made It!!!", True, self.color)
					self.screen.blit(message, (CELL_SIZE[0] , HEIGHT + (CELL_SIZE[1] * 2)))

			table.update()
			pygame.display.flip()
			self.FPS.tick(30)


if __name__ == "__main__":
	play = Main(screen)
	play.main()