""""
    THIS SCRIPT IS USING language_detector
"""
# importing the language detector
from language_detector import detect_language

# definig the function for detecting the language
# the function takes text as an argument
def detectLanguage(text):
    # detecting the language using the detect_language function
    language = detect_language(text)
    print(f'"{text}" is written in {language}')

# an infinite while while loop
while True:
    
    # this will prompt the user to enter options
    option = input('Enter 1 to detect language or 0 to exit:')
    
    if option == '1':
        # this will prompt the user to enter the text
        data = input('Enter your sentence or word here:')
        # calling the detectLanguage function
        detectLanguage(data)
    
    # if option is 0 break the loop   
    elif option == '0':
        print('Quitting........\nByee!!!')
        break
    # if option isnt 1 or 0 then its invalid 
    else:
        print('Wrong input, try again!!!')