""""
    THIS SCRIPT IS USING googletrans
"""
# importing the Translator function from googletrans
from googletrans import Translator

# initializing the translator object
translator = Translator()
# openning the txt file in read mode
sentences_file = open('sentences.txt', 'r')
# creating a list of sentences using the readlines() function
sentences = sentences_file.readlines()
# a function for detection language
def detect_langauage(sentence, n):
    """try and except block for catching exception errors"""
    # the try will run when everything is ok
    try:
        # checking if the sentence[n] exists
        if sentences[n]:
            # creating a new variable, the strip() function removes newlines
            new_sentence = sentences[n].strip('\n')
            # detecting the sentence language using the translator.detect()
            # .lang extract the language code
            detected_sentence_lang = translator.detect(new_sentence).lang 
            print(f'The language for the sentence "{new_sentence}" is {detected_sentence_lang}')
    # this will catch all the errors that occur  
    except:
        print(f'Make sure the sentence exists or you have internet connection')
           
       
print(f'You have {len(sentences)} sentences')
# this will prompt the user to enter an integer
number_of_sentence = int(input('Which sentence do you want to detect?(Provide an integer please):'))
# calling the detect_langauage function
detect_langauage(sentences_file, number_of_sentence)