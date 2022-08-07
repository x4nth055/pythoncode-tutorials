# Imports
import os
from matplotlib import pyplot as plt
import string
import numpy as np
from scipy.interpolate import make_interp_spline

# define some dictionaries
texts = {}
textlengths = {}
textwordamounts = {}

unwantedCharacters = list(string.punctuation)

# How many ranks well show
depth = 10
xAxis = [str(number) for number in range(1, depth+1)]

# Getting all files in text folder
filePaths = os.listdir('texts')

# Getting text from .txt files in folder
for path in filePaths:
    with open(os.path.join('texts', path), 'r', encoding='UTF-8') as f:
        texts[path.split('.')[0]] = f.read()


# Cleaning and counting the Text
for text in texts:
    # Remove unwanted characters from the texts
    for character in unwantedCharacters:
        texts[text] = texts[text].replace(character, '').lower()

    splittedText = texts[text].split(' ')

    # Saving the text length to show in the label of the line later
    textlengths[text] = len(splittedText)

    # Here will be the amount of occurence of each word stored
    textwordamounts[text] = {}

    # Loop through all words in the text
    for i in splittedText:

        # Add to the word at the given position if it already exists
        # Else set the amount to one essentially making a new item in the dict
        if i in textwordamounts[text].keys():
            textwordamounts[text][i] += 1
        else:
            textwordamounts[text][i] = 1

    # Sorting the dict by the values with sorted
    # define custom key so the function knows what to use when sorting
    textwordamounts[text] = dict(
        sorted(
            textwordamounts[text ].items(),
            key=lambda x: x[1],
            reverse=True)[0:depth]
        )

# Get the percentage value of a given max value
def percentify(value, max):
    return round(value / max * 100)

# Generate smooth curvess
def smoothify(yInput):

    x = np.array(range(0, depth))
    y = np.array(yInput)

    # define x as 600 equally spaced values between the min and max of original x
    x_smooth = np.linspace(x.min(), x.max(), 600) 

    # define spline with degree k=3, which determines the amount of wiggle
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    # Return the twe x and y axis
    return x_smooth, y_smooth

# Make the perfect Curve
ziffianCurveValues = [100/i for i in range(1, depth+1)]

x, y = smoothify(ziffianCurveValues)

plt.plot(x, y, label='Ziffian Curve', ls=':', color='grey')


# Plot the texts
for i in textwordamounts:
    maxValue = list(textwordamounts[i].values())[0]

    yAxis = [percentify(value, maxValue) for value in list(textwordamounts[i].values())]

    x, y = smoothify(yAxis)

    plt.plot(x, y, label=i+f' [{textlengths[i]}]', lw=1, alpha=0.5)

plt.xticks(range(0, depth), xAxis)

plt.legend()
plt.savefig('wordamounts.png', dpi=300)
plt.show()