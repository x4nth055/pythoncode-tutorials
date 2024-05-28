import pygame

class Pipe(pygame.sprite.Sprite):
	def __init__(self, pos, width, height, flip):
		super().__init__()
		self.width = width
		img_path = 'assets/terrain/pipe.png'
		self.image = pygame.image.load(img_path)
		self.image = pygame.transform.scale(self.image, (width, height))
		if flip:
			flipped_image = pygame.transform.flip(self.image, False, True)
			self.image = flipped_image
		self.rect = self.image.get_rect(topleft = pos)

	# update object position due to world scroll
	def update(self, x_shift):
		self.rect.x += x_shift

		# removes the pipe in the game screen once it is not shown in the screen anymore
		if self.rect.right < (-self.width):
			self.kill()