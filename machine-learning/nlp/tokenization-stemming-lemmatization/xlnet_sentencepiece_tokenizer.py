from transformers import XLNetTokenizer

tk = XLNetTokenize.from_pretrained('xlnet-base-cased')
f = open('reviews.txt', 'r')
for l in f:
    res = tk.tokenize(l.strip())
    print(res)
