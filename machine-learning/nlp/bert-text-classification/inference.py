from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

model_path = "20newsgroups-bert-base-uncased"
max_length = 512

def read_20newsgroups(test_size=0.2):
  dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
  documents = dataset.data
  labels = dataset.target
  return train_test_split(documents, labels, test_size=test_size), dataset.target_names


(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names)).to("cuda")
tokenizer = BertTokenizerFast.from_pretrained(model_path)

def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]

# Example #1
text = """With the pace of smartphone evolution moving so fast, there's always something waiting in the wings. 
No sooner have you spied the latest handset, that there's anticipation for the next big thing. 
Here we look at those phones that haven't yet launched, the upcoming phones for 2021. 
We'll be updating this list on a regular basis, with those device rumours we think are credible and exciting."""
print(get_prediction(text))
# Example #2
text = """
A black hole is a place in space where gravity pulls so much that even light can not get out. 
The gravity is so strong because matter has been squeezed into a tiny space. This can happen when a star is dying.
Because no light can get out, people can't see black holes. 
They are invisible. Space telescopes with special tools can help find black holes. 
The special tools can see how stars that are very close to black holes act differently than other stars.
"""
print(get_prediction(text))