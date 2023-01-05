from nltk import word_tokenize

def tokenize(file):
    tok = []
    f = open(file, 'r')
    for l in f:
        lst = word_tokenize(l)
        tok.append(lst)
    return tok

tokens = tokenize('reviews.txt')
for e in tokens:
    print(e)
