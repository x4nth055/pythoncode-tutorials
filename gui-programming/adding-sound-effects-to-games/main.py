import pygame, sys
from settings import WIDTH, HEIGHT
from world import World

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

class Main:
	def __init__(self, screen):
		self.screen = screen
		self.bg_img = pygame.image.load('assets/terrain/bg.png')
		self.bg_img = pygame.transform.scale(self.bg_img, (WIDTH, HEIGHT))
		self.FPS = pygame.time.Clock()

	def main(self):
		pygame.mixer.music.load("assets/sfx/bgm.wav")
		pygame.mixer.music.play(-1)
		pygame.mixer.music.set_volume(0.8)
		world = World(screen)
		while True:
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
			pygame.display.update()
			self.FPS.tick(60)

if __name__ == "__main__":
	play = Main(screen)
	play.main()
