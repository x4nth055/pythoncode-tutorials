# Imports
import pyinputplus as pyip
from random import choice

# Variables
questionTypes = ['+', '-', '*', '/', '**']
numbersRange = [num for num in range(1, 20)]
points = 0

# Hints
print('Round down to one Number after the Comma.')
print('When asked to press enter to continue, type stop to stop.\n')

# Game Loop
while True:
    # Deciding and generating question
    currenType = choice(questionTypes)

    promptEquation = str(choice(numbersRange)) + ' ' + currenType + ' ' + str(choice(numbersRange))
    solution = round(eval(promptEquation), 1)

    # Getting answer from User
    answer = pyip.inputNum(prompt=promptEquation + ' = ')

    # Feedback and Points
    if answer == solution:
        points += 1
        print('Correct!\nPoints: ',points)
    else:
        points -= 1
        print('Wrong!\nSolution: '+str(solution)+'\nPoints: ',points)

    # Stopping the Game
    if pyip.inputStr('Press "Enter" to continue', blank=True) == 'stop':
        break
    
    # Some Padding
    print('\n\n')
