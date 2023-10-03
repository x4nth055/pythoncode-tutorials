import pygame, time
import sys
from player import Player
from ball import Ball
from settings import WIDTH, HEIGHT, player_width, player_height

class Table:
	def __init__(self, screen):
		self.screen = screen
		self.game_over = False
		self.score_limit = 10
		self.winner = None
		self._generate_world()

		# text info
		self.font = pygame.font.SysFont('Bauhaus 93', 60)
		self.inst_font = pygame.font.SysFont('Bauhaus 93', 30)
		self.color = pygame.Color("white")

	# create and add player to the screen
	def _generate_world(self):
		self.playerA = Player(0, HEIGHT // 2 - (player_height // 2), player_width, player_height)
		self.playerB = Player(WIDTH - player_width,  HEIGHT // 2 - (player_height // 2), player_width, player_height)
		self.ball = Ball(WIDTH // 2 - player_width, HEIGHT - player_width, player_width)

	def _ball_hit(self):
		# if ball is not hit by a player and pass through table sides
		if self.ball.rect.left >= WIDTH:
			self.playerA.score += 1
			self.ball.rect.x = WIDTH // 2
			time.sleep(1)
		elif self.ball.rect.right <= 0:
			self.playerB.score += 1
			self.ball.rect.x = WIDTH // 2
			time.sleep(1)

		# if ball land in the player
		if pygame.Rect.colliderect(self.ball.rect, self.playerA.rect):
			self.ball.direction = "right"
		if pygame.Rect.colliderect(self.ball.rect, self.playerB.rect):
			self.ball.direction = "left"

	def _bot_opponent(self):
		if self.ball.direction == "left" and self.ball.rect.centery != self.playerA.rect.centery:
			if self.ball.rect.top <= self.playerA.rect.top:
				if self.playerA.rect.top > 0:
					self.playerA.move_up()
			if self.ball.rect.bottom >= self.playerA.rect.bottom:
				if self.playerA.rect.bottom < HEIGHT:
					self.playerA.move_bottom()

	def player_move(self):
		keys = pygame.key.get_pressed()

		# for bot opponent controls
		self._bot_opponent()

		# for player controls
		if keys[pygame.K_UP]:
			if self.playerB.rect.top > 0:
				self.playerB.move_up()
		if keys[pygame.K_DOWN]:
			if self.playerB.rect.bottom < HEIGHT:
				self.playerB.move_bottom()

	def _show_score(self):
		A_score, B_score = str(self.playerA.score), str(self.playerB.score)
		A_score = self.font.render(A_score, True, self.color)
		B_score = self.font.render(B_score, True, self.color)
		self.screen.blit(A_score, (WIDTH // 4, 50))
		self.screen.blit(B_score, ((WIDTH // 4) * 3, 50))

	def _game_end(self):
		if self.winner != None:
			print(f"{self.winner} wins!!")
			pygame.quit()
			sys.exit()

	def update(self):
		self._show_score()

		self.playerA.update(self.screen)		
		self.playerB.update(self.screen)

		self._ball_hit()

		if self.playerA.score == self.score_limit:
			self.winner = "Opponent"

		elif self.playerB.score == self.score_limit:
			self.winner = "You"

		self._game_end()
		self.ball.update(self.screen)