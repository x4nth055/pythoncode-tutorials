from transformers import BertTokenizer

tk = BertTokenizer.from_pretrained('bert-base-uncased')
f = open('reviews.txt', 'r')
for l in f:
    res = tk.tokenize(l.strip())
    print(res)
