from nltk import sent_tokenize

def tokenize(file):
    tok = []
    f = open(file, 'r')
    for l in f:
        lst = sent_tokenize(l)
        tok.append(lst)
    return tok

tokens = tokenize('reviews.txt')
for e in tokens:
    print(e)
