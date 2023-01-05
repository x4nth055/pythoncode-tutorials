def tokenize(file):
    tok = []
    f = open(file, 'r')
    for l in f:
        lst = l.split()
        tok.append(lst)
    return tok

tokens = tokenize('reviews.txt')
for e in tokens:
    print(e)
