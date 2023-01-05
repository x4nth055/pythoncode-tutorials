from nltk.stem.snowball import SnowballStemmer

word_lst = []
def stemmer(file):
    stm_lst = []
    stm = SnowballStemmer(language='english')
    f = open(file, 'r')
    for l in f:
        word_lst.append(l)
        w = stm.stem(str(l.strip()))
        stm_lst.append(w)
    return stm_lst

stm_lst = stemmer('reviews.txt')
for i in range(len(word_lst)):
    print(word_lst[i]+"-->"+stm_lst[i])
