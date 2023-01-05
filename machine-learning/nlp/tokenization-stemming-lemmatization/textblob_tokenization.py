from textblob import TextBlob

def tokenize(file):
    tok = []
    f = open(file, 'r')
    for l in f:
        lst = TextBlob(l).words
        tok.append(lst)
    return tok

tokens = tokenize('reviews.txt')
for e in tokens:
    print(e)
