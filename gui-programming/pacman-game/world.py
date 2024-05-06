import pygame
import time

from settings import HEIGHT, WIDTH, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED
from pac import Pac
from cell import Cell
from berry import Berry
from ghost import Ghost
from display import Display

class World:
	def __init__(self, screen):
		self.screen = screen

		self.player = pygame.sprite.GroupSingle()
		self.ghosts = pygame.sprite.Group()
		self.walls = pygame.sprite.Group()
		self.berries = pygame.sprite.Group()

		self.display = Display(self.screen)

		self.game_over = False
		self.reset_pos = False
		self.player_score = 0
		self.game_level = 1

		self._generate_world()


	# create and add player to the screen
	def _generate_world(self):
		# renders obstacle from the MAP table
		for y_index, col in enumerate(MAP):
			for x_index, char in enumerate(col):
				if char == "1":	# for walls
					self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
				elif char == " ":	 # for paths to be filled with berries
					self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
				elif char == "B":	# for big berries
					self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))

				# for Ghosts's starting position
				elif char == "s":
					self.ghosts.add(Ghost(x_index, y_index, "skyblue"))
				elif char == "p": 
					self.ghosts.add(Ghost(x_index, y_index, "pink"))
				elif char == "o":
					self.ghosts.add(Ghost(x_index, y_index, "orange"))
				elif char == "r":
					self.ghosts.add(Ghost(x_index, y_index, "red"))

				elif char == "P":	# for PacMan's starting position 
					self.player.add(Pac(x_index, y_index))

		self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]


	def generate_new_level(self):
		for y_index, col in enumerate(MAP):
			for x_index, char in enumerate(col):
				if char == " ":	 # for paths to be filled with berries
					self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 4))
				elif char == "B":	# for big berries
					self.berries.add(Berry(x_index, y_index, CHAR_SIZE // 2, is_power_up=True))
		time.sleep(2)


	def restart_level(self):
		self.berries.empty()
		[ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
		self.game_level = 1
		self.player.sprite.pac_score = 0
		self.player.sprite.life = 3
		self.player.sprite.move_to_start_pos()
		self.player.sprite.direction = (0, 0)
		self.player.sprite.status = "idle"
		self.generate_new_level()


	# displays nav
	def _dashboard(self):
		nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
		pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)
		
		self.display.show_life(self.player.sprite.life)
		self.display.show_level(self.game_level)
		self.display.show_score(self.player.sprite.pac_score)


	def _check_game_state(self):
		# checks if game over
		if self.player.sprite.life == 0:
			self.game_over = True

		# generates new level
		if len(self.berries) == 0 and self.player.sprite.life > 0:
			self.game_level += 1
			for ghost in self.ghosts.sprites():
				ghost.move_speed += self.game_level
				ghost.move_to_start_pos()

			self.player.sprite.move_to_start_pos()
			self.player.sprite.direction = (0, 0)
			self.player.sprite.status = "idle"
			self.generate_new_level()


	def update(self):
		if not self.game_over:
			# player movement
			pressed_key = pygame.key.get_pressed()
			self.player.sprite.animate(pressed_key, self.walls_collide_list)

			# teleporting to the other side of the map
			if self.player.sprite.rect.right <= 0:
				self.player.sprite.rect.x = WIDTH
			elif self.player.sprite.rect.left >= WIDTH:
				self.player.sprite.rect.x = 0

			# PacMan eating-berry effect
			for berry in self.berries.sprites():
				if self.player.sprite.rect.colliderect(berry.rect):
					if berry.power_up:
						self.player.sprite.immune_time = 150 # Timer based from FPS count
						self.player.sprite.pac_score += 50
					else:
						self.player.sprite.pac_score += 10
					berry.kill()

			# PacMan bumping into ghosts
			for ghost in self.ghosts.sprites():
				if self.player.sprite.rect.colliderect(ghost.rect):
					if not self.player.sprite.immune:
						time.sleep(2)
						self.player.sprite.life -= 1
						self.reset_pos = True
						break
					else:
						ghost.move_to_start_pos()
						self.player.sprite.pac_score += 100

		self._check_game_state()

		# rendering
		[wall.update(self.screen) for wall in self.walls.sprites()]
		[berry.update(self.screen) for berry in self.berries.sprites()]
		[ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
		self.ghosts.draw(self.screen)

		self.player.update()
		self.player.draw(self.screen)
		self.display.game_over() if self.game_over else None

		self._dashboard()

		# reset Pac and Ghosts position after PacMan get captured
		if self.reset_pos and not self.game_over:
			[ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
			self.player.sprite.move_to_start_pos()
			self.player.sprite.status = "idle"
			self.player.sprite.direction = (0,0)
			self.reset_pos = False

		# for restart button
		if self.game_over:
			pressed_key = pygame.key.get_pressed()
			if pressed_key[pygame.K_r]:
				self.game_over = False
				self.restart_level()