from tokenizers import Tokenizer

tk = Tokenizer.from_file("tokenizer-wiki.json")

f = open('reviews.txt', 'r')
for l in f:
    res = tk.encode(l.strip())
    print(res.tokens)
