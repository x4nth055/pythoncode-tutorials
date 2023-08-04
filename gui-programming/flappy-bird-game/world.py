import pygame
from pipe import Pipe
from bird import Bird
from game import GameIndicator
from settings import WIDTH, HEIGHT, pipe_size, pipe_gap, pipe_pair_sizes
import random

class World:
	def __init__(self, screen):
		self.screen = screen
		self.world_shift = 0
		self.current_x = 0
		self.gravity = 0.5
		self.current_pipe = None
		self.pipes = pygame.sprite.Group()
		self.player = pygame.sprite.GroupSingle()
		self._generate_world()
		self.playing = False
		self.game_over = False
		self.passed = True
		self.game = GameIndicator(screen)

	# creates the player and the obstacle
	def _generate_world(self):
		self._add_pipe()
		bird = Bird((WIDTH//2 - pipe_size, HEIGHT//2 - pipe_size), 30)
		self.player.add(bird)

	# adds pipe once the last pipe added reached the desired pipe horizontal spaces
	def _add_pipe(self):
		pipe_pair_size = random.choice(pipe_pair_sizes)
		top_pipe_height, bottom_pipe_height = pipe_pair_size[0] * pipe_size, pipe_pair_size[1] * pipe_size

		pipe_top = Pipe((WIDTH, 0 - (bottom_pipe_height + pipe_gap)), pipe_size, HEIGHT, True)
		pipe_bottom = Pipe((WIDTH, top_pipe_height + pipe_gap), pipe_size, HEIGHT, False)
		self.pipes.add(pipe_top)
		self.pipes.add(pipe_bottom)
		self.current_pipe = pipe_top

	# for moving background/obstacle
	def _scroll_x(self):
		if self.playing:
			self.world_shift = -6
		else:
			self.world_shift = 0

	# add gravity to bird for falling
	def _apply_gravity(self, player):
		if self.playing or self.game_over:
			player.direction.y += self.gravity
			player.rect.y += player.direction.y

	# handles scoring and collision
	def _handle_collisions(self):
		bird = self.player.sprite
		# for collision checking
		if pygame.sprite.groupcollide(self.player, self.pipes, False, False) or bird.rect.bottom >= HEIGHT or bird.rect.top <= 0:
			self.playing = False
			self.game_over = True
		else:
			# if player pass through the pipe gaps
			bird = self.player.sprite
			if bird.rect.x >= self.current_pipe.rect.centerx:
				bird.score += 1
				self.passed = True

	# updates the bird's overall state
	def update(self, player_event = None):
		# new pipe adder
		if self.current_pipe.rect.centerx  <= (WIDTH // 2) - pipe_size:
			self._add_pipe()
		
		# updates, draws pipes
		self.pipes.update(self.world_shift)
		self.pipes.draw(self.screen)

		# applying game physics
		self._apply_gravity(self.player.sprite)
		self._scroll_x()
		self._handle_collisions()

		# configuring player actions
		if player_event == "jump" and not self.game_over:
			player_event = True
		elif player_event == "restart":
			self.game_over = False
			self.pipes.empty()
			self.player.empty()
			self.player.score = 0
			self._generate_world()
		else:
			player_event = False

		if not self.playing:
			self.game.instructions()

		# updates, draws pipes
		self.player.update(player_event)
		self.player.draw(self.screen)

		self.game.show_score(self.player.sprite.score)