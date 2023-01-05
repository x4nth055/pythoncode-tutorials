import nltk
from nltk.corpus import wordnet
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

def generate_tag(w):
    t = nltk.pos_tag([w])[0][1][0].upper()
    dic = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'A': wordnet.ADJ,
        'R': wordnet.ADV
    }
    return dic.get(t, wordnet.VERB)

lem_lst = lemmatizer('reviews.txt')
for i in range(len(word_lst)):
    print(word_lst[i]+"-->"+lem_lst[i])
