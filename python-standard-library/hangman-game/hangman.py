from string import ascii_letters
import os
import random

class Hangman:

    def __init__(self):
        with open("./words.txt", "r") as file:
            words = file.read().split("\n")
            self.secret_word = random.choice(words)
            self.guessed_word = "*" * len(self.secret_word)

        self.incorrect_guess_limit = 6
        self.incorrect_guesses = 0
        self.wrong_guesses = []
        self.gallow_pieces = [
            "------",
            "|    |",
            "|    ",
            "|  ",
            "|   ",
            "|"
        ]
        self.gallow = "\n".join(self.gallow_pieces)
        self.man_pieces = [
            " \\",
            "/",
            " \\",
            " |",
            "/",
            "O",
        ]
    
    def greet_user(self):
        print("Hangman\n")
    
    def show_list_of_wrong_guesses(self):
        # show the list of wrong guesses
        print(f"Wrong guesses: {', '.join(self.wrong_guesses)}\n\n")

    def take_guess(self) -> str:
        # take user guess
        while True:
            guess = input("Guess a letter:\n>>> ")
            if len(guess) == 1 and guess in ascii_letters:
                break
            else:
                print("Invalid input")
        return guess

    def is_out_of_guesses(self) -> bool:
        # check if user is out of guesses
        return self.incorrect_guesses == self.incorrect_guess_limit

    def check_guess(self, guess_letter: str):
        # check guess, if correct, update guessed word
        # if wrong, update gallow
        if guess_letter in self.secret_word:
            self._correct_guess(guess_letter)
        else:
            self._wrong_guess(guess_letter)
    
    def _correct_guess(self, guess_letter: str):
        # find all index positions of the guess letter in the secret word
        index_positions = [index for index, item in enumerate(self.secret_word) if item == guess_letter]
        for i in index_positions:
            # update guessed word
            self.guessed_word = self.guessed_word[0:i] + guess_letter + self.guessed_word[i+1:]

    def _wrong_guess(self, guess_letter: str):
        # update gallow
        row = 2
        if self.incorrect_guesses > 0 and self.incorrect_guesses < 4:
            row = 3
        elif self.incorrect_guesses >= 4:
            row = 4
        self.gallow_pieces[row] = self.gallow_pieces[row] + self.man_pieces.pop()
        self.gallow = "\n".join(self.gallow_pieces)
        # update wrong guesses
        if guess_letter not in self.wrong_guesses:
            self.wrong_guesses.append(guess_letter)
        self.incorrect_guesses += 1

def main():
    hangman = Hangman()

    while True:
        # greet user and explain mechanics
        os.system('cls' if os.name=='nt' else 'clear')
        hangman.greet_user()
        # show gallow and the hidden word
        print(hangman.gallow, "\n")
        print("Secret word: ", hangman.guessed_word)
        # show the list of wrong guesses
        hangman.show_list_of_wrong_guesses()
        # check if user is out of guesses
        if hangman.is_out_of_guesses():
            print(f"Secret word is: {hangman.secret_word}")
            print("You lost")
            break
        elif hangman.guessed_word == hangman.secret_word:
            print("YOU WIN!!!")
            break
        else:
            # take user guess
            guess = hangman.take_guess()
            # check guess
            hangman.check_guess(guess)

if __name__ == "__main__":
    main()
