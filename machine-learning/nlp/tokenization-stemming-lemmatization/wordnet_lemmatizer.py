from nltk.stem import WordNetLemmatizer

word_lst = []
def lemmatizer(file):
    lem_lst = []
    lem = WordNetLemmatizer()
    f = open(file, 'r')
    for l in f:
        word_lst.append(l.strip())
        w = lem.lemmatize(str(l.strip()))
        lem_lst.append(w)
    return lem_lst

lem_lst = lemmatizer('reviews.txt')
for i in range(len(word_lst)):
    print(word_lst[i]+"-->"+lem_lst[i])
