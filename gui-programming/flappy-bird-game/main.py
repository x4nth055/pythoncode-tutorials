import pygame, sys
from settings import WIDTH, HEIGHT, ground_space
from world import World

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + ground_space))
pygame.display.set_caption("Flappy Bird")

class Main:
	def __init__(self, screen):
		self.screen = screen
		self.bg_img = pygame.image.load('assets/terrain/bg.png')
		self.bg_img = pygame.transform.scale(self.bg_img, (WIDTH, HEIGHT))
		self.ground_img = pygame.image.load('assets/terrain/ground.png')
		self.ground_scroll = 0
		self.scroll_speed = -6
		self.FPS = pygame.time.Clock()
		self.stop_ground_scroll = False

	def main(self):
		world = World(screen)
		while True:
			self.stop_ground_scroll = world.game_over
			self.screen.blit(self.bg_img, (0, 0))

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()

				elif event.type == pygame.KEYDOWN:
					if not world.playing and not world.game_over:
						world.playing = True
					if event.key == pygame.K_SPACE:
						world.update("jump")
					if event.key == pygame.K_r:
						world.update("restart")

			world.update()

			self.screen.blit(self.ground_img, (self.ground_scroll, HEIGHT))
			if not self.stop_ground_scroll:
				self.ground_scroll += self.scroll_speed
				if abs(self.ground_scroll) > 35:
					self.ground_scroll = 0

			pygame.display.update()
			self.FPS.tick(60)

if __name__ == "__main__":
	play = Main(screen)
	play.main()
