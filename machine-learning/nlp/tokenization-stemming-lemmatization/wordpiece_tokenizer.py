from tokenizers import BertWordPieceTokenizer

tk = BertWordPieceTokenizer("bert-word-piece-vocab.txt", lowercase=True)
f = open('reviews.txt', 'r')
for l in f:
    res = tk.encode(l.strip())
    print(res.tokens)
