# %%
# !pip install --upgrade transformers sentencepiece

# %%
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.2.0/en_core_web_trf-3.2.0-py3-none-any.whl

# %%
# !python -m spacy download en_core_web_sm

# %%
import spacy
from transformers import *

# %%
# sample text from Wikipedia
text = """
Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
Einstein is best known for developing the theory of relativity, but he also made important contributions to the development of the theory of quantum mechanics.
Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of Württemberg) the following year. 
In 1897, at the age of 17, he enrolled in the mathematics and physics teaching diploma program at the Swiss Federal polytechnic school in Zürich, graduating in 1900
"""

# %%
# load BERT model fine-tuned for Named Entity Recognition (NER)
ner = pipeline("ner", model="dslim/bert-base-NER")

# %%
# perform inference on the transformer model
doc_ner = ner(text)
# print the output
doc_ner

# %%
def get_entities_html(text, ner_result, title=None):
  """Returns a visual version of NER with the help of SpaCy"""
  ents = []
  for ent in ner_result:
    e = {}
    # add the start and end positions of the entity
    e["start"] = ent["start"]
    e["end"] = ent["end"]
    # add the score if you want in the label
    # e["label"] = f"{ent["entity"]}-{ent['score']:.2f}"
    e["label"] = ent["entity"]
    if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
      # if the current entity is shared with previous entity
      # simply extend the entity end position instead of adding a new one
      ents[-1]["end"] = e["end"]
      continue
    ents.append(e)
  # construct data required for displacy.render() method
  render_data = [
    {
      "text": text,
      "ents": ents,
      "title": title,
    }
  ]
  return spacy.displacy.render(render_data, style="ent", manual=True, jupyter=True)

# %%
# get HTML representation of NER of our text
get_entities_html(text, doc_ner)

# %%
# load roberta-large model
ner2 = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")

# %%
# perform inference on this model
doc_ner2 = ner2(text)

# %%
# get HTML representation of NER of our text
get_entities_html(text, doc_ner2)

# %%
# load yet another roberta-large model
ner3 = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")

# %%
# perform inference on this model
doc_ner3 = ner3(text)

# %%
# get HTML representation of NER of our text
get_entities_html(text, doc_ner3)

# %%
# load the English CPU-optimized pipeline
nlp = spacy.load("en_core_web_sm")

# %%
# predict the entities
doc = nlp(text)

# %%
# display the doc with jupyter mode
spacy.displacy.render(doc, style="ent", jupyter=True)

# %%
# load the English transformer pipeline (roberta-base) using spaCy
nlp_trf = spacy.load('en_core_web_trf')

# %%
# perform inference on the model
doc_trf = nlp_trf(text)
# display the doc with jupyter mode
spacy.displacy.render(doc_trf, style="ent", jupyter=True)


