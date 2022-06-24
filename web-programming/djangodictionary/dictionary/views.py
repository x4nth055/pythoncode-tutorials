# importing the render function from django.shortcuts
# the render function renders templates
from django.shortcuts import render
# importing the PyDictionary library
from PyDictionary import PyDictionary

import pyttsx3


# this is the view that will render the index page
def homeView(request):
    return render(request, 'dictionary/index.html')


# this is the view that will render search page
def searchView(request):
    # capturing the word from the form via the name search
    word = request.GET.get('search')
    # creating a dictionary object
    dictionary = PyDictionary()
    # passing a word to the dictionary object
    meanings = dictionary.meaning(word)
    # getting a synonym and antonym  
    synonyms = dictionary.synonym(word)
    antonyms = dictionary.antonym(word)
    # bundling all the variables in the context  
    context = {
            'word': word,
            'meanings':meanings,
            'synonyms':synonyms,
            'antonoyms':antonyms
        }
    return render(request, 'dictionary/search.html', context)


