# Import pyinputplus for choice inputs and os to clear the console.
import pyinputplus
import os
import json

# setting up some variables
currentKey = '0'
currentKeys = []
itemAlreadyAdded = False

# Get the Story Prompts
# A dictionary is used because we dont want to allow
# duplicate keys
with open('story.json', 'r') as f:
    storyPrompts = json.load(f)

inventory = {
    'banana(s)': 0,
    'clock(s)': 2,
    'swords(s)': 0,
}

# Check if the prompts are valid
for prompt in storyPrompts:
    promptText, keys, *_ = storyPrompts[prompt]

    # Add ":" at the end of the prompt Text
    if not promptText.endswith(':'):
        storyPrompts[prompt][0] = promptText + ': '

    # Check if the keys are strings, if not transform them
    storyPrompts[prompt][1] = [str(i) for i in keys]


# Giving the user some instructions.
print('Type in the number of the prompt or -i to view your inventory ... have fun.')

# Prompt Loop
while True:
    # Clearing the Console on all platforms
    os.system('cls' if os.name == 'nt' else 'clear')
    # Get the current prompt all its associated data
    currentPrompt, currentKeys, _, action = storyPrompts[currentKey]
    # Finish the Adventure when the next keys list contains the string 'end'
    if 'end' in currentKeys:
        break
    # Look for inventory Changes
    if not itemAlreadyAdded:
        if 'minus' in action:
            inventory[action.split('-')[1]+'(s)'] -= 1
        if 'plus' in action:
            inventory[action.split('-')[1]+'(s)'] += 1
    # Add Option Descriptions to the current Prompt with their number
    for o in currentKeys:

        invalidOption = False

        thisaction = storyPrompts[o][3]
        if 'minus' in thisaction:
            item = storyPrompts[o][3].split('-')[1]+'(s)'
            if inventory[item] == 0:
                print(storyPrompts[o][3].split('-')[1]+'(s)')
                invalidOption = True

        if not invalidOption:
            currentPrompt += f'\n{o}. {storyPrompts[o][2]}'


    currentPrompt += '\n\nWhat do you do? '

    # Get the input from the user, only give them the keys as a choice so they dont
    # type in something invalid.
    userInput = pyinputplus.inputChoice(choices=(currentKeys + ['-i']), prompt=currentPrompt)

    # Printing out the inventory if the user types in -i
    if '-i' in userInput:
        print(f'\nCurrent Inventory: ')

        for i in inventory:
            print(f'{i} : {inventory[i]}')

        print ('\n')

        input('Press Enter to continue ... ')

        itemAlreadyAdded = True

        continue
    else:
        itemAlreadyAdded = False

    currentKey = userInput

# Printing out the last prompt so the user knows what happened to him.
print(storyPrompts[currentKey][0])
print('\nStory Finished ...')
