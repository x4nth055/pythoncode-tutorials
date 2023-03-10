import pygame
from pygame.locals import *
import random
from string import ascii_letters

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((400, 500))
pygame.display.set_caption("Hangman")

class Hangman():
    def __init__(self):
        with open("./words.txt", "r") as file:
            # picks secret word
            words = file.read().split("\n")
            self.secret_word = random.choice(words)
            # passing secret word's length for making letter blanks
            self.guessed_word = "*" * len(self.secret_word)
        self.wrong_guesses = []
        self.wrong_guess_count = 0
        self.taking_guess = True
        self.running = True

        self.background_color = (155, 120, 70)
        self.gallow_color = (0,0,0)
        self.body_color = (255,253,175)

        self.font = pygame.font.SysFont("Courier New", 20)
        self.FPS = pygame.time.Clock()


    # draws the gallow
    def _gallow(self):
        stand = pygame.draw.rect(screen, self.gallow_color, pygame.Rect(75, 280, 120, 10))
        body = pygame.draw.rect(screen, self.gallow_color, pygame.Rect(128, 40, 10, 240))
        hanger = pygame.draw.rect(screen, self.gallow_color, pygame.Rect(128, 40, 80, 10))
        rope = pygame.draw.rect(screen, self.gallow_color, pygame.Rect(205, 40,10, 30))


    # draw man's body parts for every wrong guess
    def _man_pieces(self):
        if self.wrong_guess_count == 1:
            head = pygame.draw.circle(screen, self.body_color, [210, 85], 20, 0)
        elif self.wrong_guess_count == 2:
            body = pygame.draw.rect(screen, self.body_color, pygame.Rect(206, 105, 8, 45))
        elif self.wrong_guess_count == 3:
            r_arm = pygame.draw.line(screen, self.body_color, [183, 149], [200, 107], 6)
        elif self.wrong_guess_count == 4:
            l_arm = pygame.draw.line(screen, self.body_color, [231, 149], [218, 107], 6),
        elif self.wrong_guess_count == 5:
            r_leg = pygame.draw.line(screen, self.body_color, [189, 198], [208, 148], 6),
        elif self.wrong_guess_count == 6:
            l_leg = pygame.draw.line(screen, self.body_color, [224, 198], [210, 148], 6)


    def _right_guess(self, guess_letter):
        index_positions = [index for index, item in enumerate(self.secret_word) if item == guess_letter]
        for i in index_positions:
            self.guessed_word = self.guessed_word[0:i] + guess_letter + self.guessed_word[i+1:]
        # stacks a layer of color on guessed word to hide multiple guessed_word stack
        screen.fill(pygame.Color(self.background_color), (10, 370, 390, 20))


    def _wrong_guess(self, guess_letter):
        self.wrong_guesses.append(guess_letter)
        self.wrong_guess_count += 1
        self._man_pieces()


    def _guess_taker(self, guess_letter):
        if guess_letter in ascii_letters:
            if guess_letter in self.secret_word and guess_letter not in self.guessed_word:
                self._right_guess(guess_letter)
            elif guess_letter not in self.secret_word and guess_letter not in self.wrong_guesses:
                self._wrong_guess(guess_letter)


    def _message(self):
        # win situation
        if self.guessed_word == self.secret_word:
            self.taking_guess = False
            screen.fill(pygame.Color(0,0,79), (40, 218, 320, 30))
            message = self.font.render("YOU WIN!!", True, (255,235,0))
            screen.blit(message,(152,224))

        # lose situation
        elif self.wrong_guess_count == 6:
            self.taking_guess = False
            screen.fill(pygame.Color("grey"), (40, 218, 320, 30))
            message = self.font.render("GAME OVER YOU LOSE!!", True, (150,0,10))
            screen.blit(message,(78,224))
            # shows the secret word if the player lose
            word = self.font.render(f"secret word: {self.secret_word}", True, (255,255,255))
            screen.blit(word,(10,300))

        # removes the instruction message if not taking guesses anymore
        if not self.taking_guess:
            screen.fill(pygame.Color(self.background_color), (35, 460, 390, 20))


    def main(self):
        # game's main components (no need to update)
        screen.fill(self.background_color)
        self._gallow()
        instructions = self.font.render('Press any key to take Guess', True, (9,255,78))
        screen.blit(instructions,(35,460))

        while self.running:
            # shows the guessed word in the game window
            guessed_word = self.font.render(f"guessed word: {self.guessed_word}", True, (0,0,138))
            screen.blit(guessed_word,(10,370))
            # shows the wrong guesses in the game window
            wrong_guesses = self.font.render(f"wrong guesses: {' '.join(map(str, self.wrong_guesses))}", True, (125,0,0))
            screen.blit(wrong_guesses,(10,420))

            # checking game state
            self._message()
        
            for self.event in pygame.event.get():
                if self.event.type == pygame.QUIT:
                    self.running = False

                # manages keys pressed
                elif self.event.type == pygame.KEYDOWN:
                    if self.taking_guess:
                        self._guess_taker(self.event.unicode)

            pygame.display.flip()
            self.FPS.tick(60)

        pygame.quit()


if __name__ =="__main__":
    h = Hangman()
    h.main()
