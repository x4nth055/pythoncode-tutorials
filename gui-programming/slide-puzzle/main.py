import pygame

from frame import Frame
from game import Game

pygame.init()
pygame.font.init()

class Puzzle:
	def __init__(self, screen):
		self.screen = screen
		self.running = True
		self.FPS = pygame.time.Clock()
		self.is_arranged = False
		self.font = pygame.font.SysFont("Courier New", 33)
		self.background_color = (255, 174, 66)
		self.message_color = (17, 53, 165)

	def _draw(self, frame):
		frame.draw(self.screen)
		pygame.display.update()

	def _instruction(self):
		instructions = self.font.render('Use Arrow Keys to Move', True, self.message_color)
		screen.blit(instructions,(5,460))

	def main(self, frame_size):
		self.screen.fill("white")
		frame = Frame(frame_size)
		game = Game()
		self._instruction()
		while self.running:

			if game.is_game_over(frame):
				self.is_arranged = True
				game.message(self.screen)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False

				if event.type == pygame.KEYDOWN:
					if not self.is_arranged:
						if game.arrow_key_clicked(event):
							frame.handle_click(event)

			self._draw(frame)
			self.FPS.tick(30)
	
		pygame.quit()


if __name__ == "__main__":
	window_size = (450, 500)
	screen = pygame.display.set_mode(window_size)
	pygame.display.set_caption("Slide Puzzle")

	game = Puzzle(screen)
	game.main(window_size[0])