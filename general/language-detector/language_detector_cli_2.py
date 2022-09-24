""""
    THIS SCRIPT IS USING langid
"""
import langid

# openning the txt file in read mode
sentences_file = open('sentences.txt', 'r')

# creating a list of sentences using the readlines() function
sentences = sentences_file.readlines()

# looping through all the sentences in thesentences.txt file
for sentence in sentences:
    # detecting the languages for the sentences
    lang = langid.classify(sentence)
    # formatting the sentence by removing the newline characters
    formatted_sentence = sentence.strip('\n')
   
    print(f'The sentence "{formatted_sentence}" is in {lang[0]}')